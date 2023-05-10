import torch.nn as nn
import copy
from transformers import T5ForConditionalGeneration #, MT5ForConditionalGeneration)
from transformers.models.t5.modeling_t5 import T5Stack
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch
import numpy as np
import os
from math import inf

import utils.tree as u_trees
from model.gate import HighwayGateLayer

def pool(h, mask, type='max'):
    if type == 'max':
        # masked_fill
        h = h.masked_fill(mask, -inf)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        in_hidden_size = config.hidden_size

        self.pool_type = config.pooler

        # output MLP layers
        layers = [nn.Linear(in_hidden_size, config.hidden_size), nn.Tanh()]
        for _ in range(config.mlp_layers - 1):
            layers += [nn.Linear(config.hidden_size, config.hidden_size),
                       nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, hidden_states, token_mask=None):
        """ 根据token_mask进行pool操作, 然后过mlp层. 
        para
            token_mask: mask矩阵
        """
        if token_mask is None:
            pool_mask = torch.ones_like(hidden_states).bool()
        else:
            pool_mask = token_mask.eq(0).unsqueeze(2)
        h_out = pool(hidden_states,
                     pool_mask,
                     type=self.pool_type)

        pooled_output = self.out_mlp(h_out)
        return pooled_output
    
class T5BaseFullModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        
        # config
        self.config = config
        self.tokenizer = tokenizer
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_kldiv = nn.KLDivLoss(reduction='batchmean')
        
        # self.model = MT5ForConditionalGeneration.from_pretrained(config.bert_model)
        self.model:T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(config.base_model)
        
        self.model.config.eos_token_id=0    # 如何设置?
        
        self.t5_config = self.model.config
        self.encoder:T5Stack = self.model.encoder
        
        # text2spo
        self.triples_decoder:T5Stack = self.model.decoder
        self.triples_lm_head = self.model.lm_head
        # spo2text
        self.text_decoder:T5Stack = copy.deepcopy(self.triples_decoder)
        self.text_lm_head = copy.deepcopy(self.triples_lm_head)
        
        self.pooler = Pooler(config)
        # 
        self.gate = HighwayGateLayer(config.hidden_size)

    @staticmethod
    def inputs_to_tree_reps(dep_head, dep_rel, maxlen):
        # maxlen = max(seq_len)
        seq_len = [len(i) for i in dep_head]
        trees = [u_trees.head_to_tree(dep_head[i], dep_rel[i], 
                              # self.config.syntax['prune_k'],
                              ) for i in range(len(seq_len))]

        # Making "self_loop=True" as adj will be used as a masking matrix during graph attention
        adj_matrix_list, dep_rel_matrix_list = [], []
        for tree in trees:
            adj_matrix, dep_rel_matrix = u_trees.tree_to_adj(maxlen, tree,
                                                     directed=False,
                                                     self_loop=True) #self.config.syntax['adj_self_loop'])
            adj_matrix = adj_matrix.reshape(1, maxlen, maxlen)
            adj_matrix_list.append(adj_matrix)

            dep_rel_matrix = dep_rel_matrix.reshape(1, maxlen, maxlen)
            dep_rel_matrix_list.append(dep_rel_matrix)

        """ debug: 检察att矩阵? 只有一个位置是1? 原因是dataloader中 _wp_aligned_dep_parse 函数写错了. #done """
        batch_adj_matrix = torch.from_numpy(np.concatenate(adj_matrix_list, axis=0))
        batch_dep_rel_matrix = torch.from_numpy(np.concatenate(dep_rel_matrix_list, axis=0))
        return Variable(batch_adj_matrix), Variable(batch_dep_rel_matrix)


    def forward(self, input_ids, attention_mask, 
                spo_input_ids=None, spo_attention_mask=None, 
                dep_head=None, dep_rel=None, 
                mode='train'):
        """ 模型
        训练过程: 返回 loss
        验证过程: 返回 生成的文本
        inputs:
            input_ids, attention_mask: (batch_size, seq_len)
        """
        if mode=='train':
            adj_matrix, dep_rel_matrix = self.inputs_to_tree_reps(dep_head, dep_rel, maxlen=self.config.max_len)
            adj_matrix = adj_matrix.to(input_ids.device)
            dep_rel_matrix = dep_rel_matrix.to(input_ids.device)
            
            # (bs, seqlen, h)
            h = None
            if self.config.fuse_mode=='s-adj':
                # TODO: 实现乘法mask公式, 需要修改 modelling_t5.py
                h = self.encoder(input_ids=input_ids, attention_mask=adj_matrix)[0]
            elif self.config.fuse_mode=='s-bert':
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0] # , output_attentions=True
            elif self.config.fuse_mode=='gate':
                h_syntax = self.encoder(input_ids=input_ids, attention_mask=adj_matrix)[0]
                h_bert = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                h = self.gate(h_syntax, h_bert)
            else: raise ValueError
            
            triples_decoder_input = self.shift_tokens_right(spo_input_ids)
            triples_decoder_output = self.triples_decoder(input_ids=triples_decoder_input, encoder_hidden_states=h)
            triples_logits = self.triples_lm_head(triples_decoder_output[0])
            
            text_encoder_output = self.encoder(input_ids=spo_input_ids, attention_mask=spo_attention_mask)
            text_decoder_input = self.shift_tokens_right(input_ids)
            text_decoder_output = self.text_decoder(input_ids=text_decoder_input, encoder_hidden_states=text_encoder_output[0])
            text_logits = self.text_lm_head(text_decoder_output[0])
            
            origin_pooled_output = self.pooler(h, attention_mask)
            gen_pooled_output = self.pooler(text_decoder_output[0], attention_mask)
            consistency_loss = self.loss_kldiv(torch.log_softmax(origin_pooled_output, dim=-1), torch.softmax(gen_pooled_output, dim=-1))
            
            triples_loss = self.loss_fct(
                triples_logits.view(-1,self.t5_config.vocab_size), spo_input_ids.view(-1))
            text_loss = self.loss_fct(
                text_logits.view(-1,self.t5_config.vocab_size), input_ids.view(-1))
            
            loss = triples_loss
            if self.config.dual_mode == 'consist':
                loss += consistency_loss
            elif self.config.dual_mode == 'gen':
                loss += text_loss
            else: raise ValueError
            return {
                'triple_loss': triples_loss,
                'text_loss': text_loss,
                'consist_loss': consistency_loss,
                'loss': loss
            }
        
        elif mode=='eval':
            output_sequences = self.model.generate(input_ids=input_ids, 
                                                   num_beams=5, do_sample=False,
                                                   attention_mask=attention_mask, 
                                                   max_length=128)
            outputs = self.tokenizer.batch_decode(output_sequences, 
                                                  skip_special_tokens=True
                                                  )
            return outputs
    
    def shift_tokens_right(self,input_ids: torch.Tensor):
        """ Shift input ids one token to the right. """
        return self.model._shift_right(input_ids)



