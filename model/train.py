# _*_ coding:utf-8 _*_
import codecs
import json
import logging
import sys
import time
from warnings import simplefilter
# import prettytable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from prettytable import PrettyTable
import shutil
from functools import lru_cache
from transformers import AdamW

from model.model import T5BaseFullModel
import utils.util_bert as util_bert

simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, data_loaders, examples, tokenizer, writer):
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        # self.n_gpu = torch.cuda.device_count()
        

        self.model:T5BaseFullModel = T5BaseFullModel(args, tokenizer)

        self.model.to(self.device)
        if args.train_mode == "predict":
            self.resume(args)

        train_dataloader, dev_dataloader = data_loaders
        train_eval, dev_eval = examples
        self.eval_file_choice = {
            "train": train_eval,
            "dev": dev_eval,
        }
        self.data_loader_choice = {
            "train": train_dataloader,
            "dev": dev_dataloader,
        }
        self.writer = writer
        # TODO 稍后要改成新的优化器，并加入梯度截断
        self.optimizer = self.set_optimizer(args, self.model,
                                            train_steps=(int(
                                                len(train_eval) / args.train_batch_size) + 1) * args.epoch_num)

    def set_optimizer(self, args, model, train_steps=None):
        """ 设置优化器
        通过 args 参数控制
        """
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                            #  warmup=args.warmup_proportion,
                            #  t_total=train_steps
                             )
        return optimizer


    def train(self, args):
        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        g_steps = 0
        for epoch in range(int(args.epoch_num)):
            t_epoch_start = time.time()
            
            batch_loss = 0.0
            batch_cnt = 0

            train_dataloader = self.data_loader_choice[u"train"]
            pbar = tqdm(train_dataloader, total=len(train_dataloader), # ncols=100, unit_scale=False, 
                                mininterval=5,
                                desc=f'training at epoch : {epoch} ', # leave=False, file=sys.stdout
                                )
            for step, batch in enumerate(pbar):
                _res = self._forward(batch)
                loss, triple_loss, text_loss, consist_loss = _res['loss'], _res['triple_loss'], _res['text_loss'], _res['consist_loss']
                loss, triple_loss, text_loss, consist_loss = loss.item(), triple_loss.item(), text_loss.item(), consist_loss.item()
                
                _batch_cnt = len(batch[0])
                batch_loss += loss * _batch_cnt
                batch_cnt += _batch_cnt
                
                g_steps += 1
                self.writer.add_scalar('loss/sum', loss, global_step=g_steps)
                self.writer.add_scalar('loss/triple', triple_loss, global_step=g_steps)
                self.writer.add_scalar('loss/text', text_loss, global_step=g_steps)
                self.writer.add_scalar('loss/consist', consist_loss, global_step=g_steps)

                pbar.set_postfix({'loss': batch_loss / batch_cnt})
            
            t_epoch_train_end = time.time()
            logger.info(f"epoch {epoch} training finished, time consumed: {t_epoch_train_end-t_epoch_start:.4f}")


            if epoch % 10 == 0:
                res_dev = self.eval_data_set("dev", outfn=f"{args.log_dir}/dev_epoch_{epoch}.log")
                
                res_table = PrettyTable()
                res_table.field_names = ["Epoch", "F1", "P", "R"]
                res_table.add_row([epoch, res_dev['f1'], res_dev['precision'], res_dev['recall']])

                for _m in ['f1', 'recall', 'precision']:
                    self.writer.add_scalar(f'eval/{_m}', res_dev[_m], epoch)
                
                if res_dev['f1'] > best_f1:
                    best_f1 = res_dev['f1']
                    logging.info("** ** * Saving fine-tuned model ** ** * ")
                    model_to_save = self.model.module if hasattr(self.model,
                                                                'module') else self.model  # Only save the model it-self
                    output_model_file = f"{args.ckpt_dir}/pytorch_model.bin"
                    torch.save(model_to_save.state_dict(), str(output_model_file))
                    patience_stop = 0
                else:
                    patience_stop += 1
                if patience_stop >= args.patience_stop:
                    logger.info(f"Eary stop at epoch {epoch}, with patience {patience_stop}!")
                    return

    def resume(self, args):
        resume_model_file = f"{args.ckpt_dir}/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def _forward(self, batch, chosen=u'train', eval=False):
        batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
        if not eval:
            p_ids, input_ids, attention_mask, dep_head, dep_rel, spo_input_ids, spo_attention_mask = batch
            _ret = self.model(input_ids=input_ids,attention_mask=attention_mask,spo_input_ids=spo_input_ids,spo_attention_mask=spo_attention_mask,
                              dep_head=dep_head, dep_rel=dep_rel)
            loss = _ret['loss']
            # if self.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu.

            self.optimizer.zero_grad()
            loss.backward()
            # loss = loss.item()
            self.optimizer.step()
            return _ret
        else:
            p_ids, input_ids, attention_mask, dep_head, dep_rel = batch
            # eval_file = self.eval_file_choice[chosen]
            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, mode='eval', 
                                #  dep_head=dep_head, dep_rel=dep_rel
                                 )
            return outputs

    def eval_data_set(self, chosen="dev", outfn=None):
        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        preds = []

        t_eval_s = time.time()
        with torch.no_grad():
            pbar = tqdm(data_loader, mininterval=5, total=len(data_loader), 
                        #leave=False, file=sys.stdout
                        desc=f'evaling at [{chosen}]' )
            for _, batch in enumerate(pbar):
                o = self._forward(batch, chosen, eval=True)
                preds.extend(o)
        t_eval_e = time.time()
        logger.info(f'eval {chosen} finished, time took : {t_eval_e-t_eval_s:.4f} sec')
        
        res = _evaluate(eval_file, preds, chosen, fuzzy=self.args.eval_fuzzy)
        self.model.train()
        
        if outfn:
            with open(outfn, 'w') as f:
                f.write(f"F1: {res['f1']:.4f}, Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}\n\n")
                for idx, (ex, spo_pred) in enumerate(zip(eval_file, preds)):
                    spo_gold = [util_bert.spo2text(p) for p in ex.spo]
                    spo_pred = util_bert.split2spo_text(spo_pred)
                    if spo_pred!=spo_gold:
                        f.write(f"{idx}\t{''.join(ex.token)}\n{spo_gold}\n{spo_pred}\n\n")

        return res

    def predict_data_set(self, chosen="dev"):
        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], [], {}] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self._forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        

    def show(self, chosen="dev"):
        """  """
        self.model.eval()
        answer_dict = {}

        data_loader = self.data_loader_choice[chosen]
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                loss, answer_dict_ = self._forward(batch, chosen, eval=True)
                answer_dict.update(answer_dict_)

def _evaluate(eval_file, preds, chosen='train', fuzzy=False):
    nsample = 10
    df = pd.DataFrame({
        'golden': [[util_bert.spo2text(p) for p in e.spo] for e in eval_file[:nsample]],
        'pred': [util_bert.split2spo_text(p) for p in preds[:nsample]]
    })
    logging.info(f'sample result: {df}')
    tp, fp, fn = 0, 0, 0
    for ex, spo_pred in zip(eval_file, preds):
        spo_list = ex.spo
        spo_gold = [util_bert.spo2text(s) for s in spo_list]
        spo_pred = util_bert.split2spo_text(spo_pred)
        
        tp_tmp, fp_tmp, fn_tmp = calculate_metric(spo_gold, spo_pred, fuzzy=fuzzy)
        tp += tp_tmp
        fp += fp_tmp
        fn += fn_tmp

    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f = 2 * p * r / (p + r) if p + r != 0 else 0

    logging.info('============================================')
    logging.info(f"{chosen}/em: {tp},\tpre&gold: {tp + fp}\t{tp + fn}")
    logging.info(f"{chosen}/f1: {100*f}, \tPrecision: {100*p},\tRecall: {100*r}")
    return {'f1': f, "recall": r, "precision": p}

def calculate_metric(spo_list_gt, spo_list_predict, fuzzy=False):
    # calculate golden metric precision, recall and f1
    if fuzzy:
        tp = calc_score(spo_list_gt, spo_list_predict)
        fp = len(spo_list_predict) - tp
        fn = len(spo_list_gt) - tp
    else:
        tp = 0
        for p in spo_list_predict:
            tp += (p in spo_list_gt)
        fp = len(spo_list_predict) - tp
        fn = len(spo_list_gt) - tp
    return tp, fp, fn

def calc_match(label, pred):
    m,n = len(label),len(pred)
    @lru_cache(None)
    def f(i,j):
        if i==m and j==n: return 0
        if i==m: return n-j
        if j==n: return m-i
        if label[i]==pred[j]: return f(i+1,j+1)
        else: return min(f(i+1,j),f(i,j+1),f(i+1,j+1))+1
    return 1 - f(0,0) / max(m,n)

def calc_score(labels, preds):
    if len(labels)==0 or len(preds)==0:
        return 0
    s = 0
    for p in preds:
        s += max([calc_match(l,p) for l in labels])
    return s