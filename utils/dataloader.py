import json
import logging
import random
import re
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils.util_file as utils_file
import utils.util_io as utils_io
import utils.constant as constant
import utils.util_bert as util_bert



class Example(object):
    def __init__(self, token, head, deprel, spo):
        self.token = token
        self.head = head
        self.deprel = deprel
        self.spo = spo

class Reader(object):
    def __init__(self,):
        pass

    def read_examples(self, filename, data_type):
        """ 读取接口,  """
        logging.info("Generating {} examples...".format(data_type))
        return self._read(filename, data_type)

    def _read(self, filename, data_type):
        examples = []
        for line in tqdm(utils_io.LoadJsonl(filename)):
             examples.append(Example(token=line['token'], head=line['head'], deprel=line['deprel'], spo=line['label']))

        logging.info("{} total size is  {} ".format(data_type, len(examples)))
        return examples



class Feature(object):
    def __init__(self, max_len, tokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __call__(self, examples, data_type):
        return self.convert_examples_to_bert_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):
        logging.info("convert {}  examples to features .".format(data_type))

        logging.info("Built instances is Completed")
        return SPODataset(examples, data_type=data_type,
                          tokenizer=self.tokenizer, max_len=self.max_len)


class SPODataset(Dataset):
    def __init__(self, data, data_type, tokenizer=None, max_len=128, mask_padding_with_zero=True, pad_token=0):
        super(SPODataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.q_ids = [f[0] for f in data]
        # self.features = [f[1] for f in data]
        self.is_train = True if data_type == 'train' else False
        self.mask_padding_with_zero = mask_padding_with_zero
        self.pad_token = tokenizer.pad_token_id
        data = self.convert_examples_to_features(data, tokenizer)
        self.q_ids = list(range(len(data)))
        self.features = data

    @staticmethod
    def _tokenize_with_bert(sequence, tokenizer):
        word_idxs, bert_tokens, subword_token_len = [], [], []
        idx = 0
        for s in sequence:
            tokens = tokenizer.tokenize(s)
            subword_token_len.append(len(tokens))
            word_idxs += [idx]
            bert_tokens += tokens
            idx += len(tokens)
        return bert_tokens, word_idxs, subword_token_len

    @staticmethod
    def _handle_long_sent_parses(dep_head, dep_rel, length):
        dep_rel = dep_rel[:length]
        dep_head, truncated_head = dep_head[:length], dep_head[length:]

        # Check if the ROOT lies in the remaining part of the truncated sequence
        # And if so, make the last token in the truncated sequence as ROOT
        is_root = [True for x in truncated_head if x == 0]
        if is_root:
            dep_head[-1] = 0

        # Assert that there is only one ROOT in the parse tree
        dep_root_ = [i for i, x in enumerate(dep_head) if x == 0]
        assert len(dep_root_) == 1

        # If head word index is greater than max_length then connect the word to itself
        for i, head_word_index in enumerate(dep_head):
            if head_word_index > len(dep_head):
                dep_head[i] = i + 1
                # dep_head[i] = length

        return dep_head, dep_rel

    @staticmethod
    def _wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel):
        wp_dep_head, wp_dep_rel = [], []
        for i, (idx, slen) in enumerate(zip(word_idxs, subword_token_len)):
            rel = dep_rel[i]
            wp_dep_rel.append(rel)
            head = dep_head[i]
            # ROOT token in the parse tree
            if head == 0:
                # This index is what the other words will refer to as the ROOT word
                # root_pos = i + 1
                wp_dep_head.append(0)
            else:
                # Stanford Dependency Parses are offset by 1 as 0 is the ROOT token
                # if head < len(word_idxs):
                assert head <= len(word_idxs)
                # 转化为subword的index!!!
                new_pos = word_idxs[head-1] + 1
                wp_dep_head.append(new_pos)

            for _ in range(1, slen):
                # Add special DEP-REL for the subwords
                wp_dep_rel.append(constant.DEPREL_TO_ID['subtokens'])
                # Add special directed edges from the first subword to the next subword
                wp_dep_head.append(idx)
        return wp_dep_head, wp_dep_rel

    def convert_examples_to_features(self, examples, tokenizer):
        data = []
        for (ex_index, example) in enumerate(examples):
            raw_tokens = example.token
            dep_head, dep_rel = example.head, example.deprel
            dep_rel = map_to_ids(dep_rel, constant.DEPREL_TO_ID)
            
            tokens, word_idxs, subword_token_len = self._tokenize_with_bert(raw_tokens, tokenizer)
            
            while len(tokens) > self.max_len:
                # tokens = tokens[:(max_len - 2)]
                ratio = len(tokens) / len(raw_tokens)
                est_len = int((self.max_len - 2) / ratio)
                raw_tokens = raw_tokens[:est_len]
                # Takes care of edge cases for dependency head and dependency relation when truncating a sequence
                dep_head, dep_rel = self._handle_long_sent_parses(dep_head, dep_rel, est_len)
                tokens, word_idxs, subword_token_len = self._tokenize_with_bert(raw_tokens, tokenizer)


            dep_head, dep_rel = self._wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel)
            assert len(dep_rel) == len(tokens)
            assert len(dep_head) == len(tokens)
            assert max(dep_head) <= len(tokens)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token] * padding_length)
            # dep_rel = dep_rel + ([self.pad_token] * (self.max_len - len(dep_rel)))
            input_mask = input_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
            
            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            data.append((
                ex_index, example.spo,
                input_ids, input_mask, dep_head, dep_rel
            ))
            
        return data

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.features[index]

    def _create_collate_fn(self):
        def collate(examples):
            p_ids, ds = zip(*examples)
            
            spos = [d[1] for d in ds]
            # outputs =[";".join('+'.join(spo) for spo in spos_single) for spos_single in spos]
            outputs = [util_bert.spo_list2text(spos_single) for spos_single in spos]
            _output = self.tokenizer(outputs, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
            spo_input_ids = _output.input_ids
            spo_attention_mask = _output.attention_mask
            
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            
            input_ids = torch.tensor([i[2] for i in ds], dtype=torch.long)
            attention_mask =torch.tensor([i[3] for i in ds], dtype=torch.long)
            
            dep_head = [i[4] for i in ds]
            dep_rel = [i[5] for i in ds]

            if not self.is_train:
                return p_ids, input_ids, attention_mask, dep_head, dep_rel
            else:
                return p_ids, input_ids, attention_mask, dep_head, dep_rel, spo_input_ids, spo_attention_mask
        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

def map_to_ids(tokens, vocab):
    assert all([t in vocab for t in tokens]), f"Unknown tokens: {set(tokens) - set(vocab)}"
    # ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    ids = [vocab[t] for t in tokens]
    return ids

def build_dataset(args, reader: Reader, tokenizer, debug=False, force_reload=True):
    logging.info("** ** * bulid dataset ** ** * ")
    train_src = f"{args.data_dir}/train.json"
    dev_src = f"{args.data_dir}/dev.json"
    
    train_examples_file = args.data_cache_dir + "/train-examples.pkl"
    dev_examples_file = args.data_cache_dir + "/dev-examples.pkl"

    if force_reload or (not os.path.exists(train_examples_file)):
        train_examples = reader.read_examples(train_src, data_type='train')
        dev_examples = reader.read_examples(dev_src, data_type='dev')
        utils_file.save(train_examples_file, train_examples, message="train examples")
        utils_file.save(dev_examples_file, dev_examples, message="dev examples")
    else:
        logging.info('loading train data_cache_dir {}'.format(train_examples_file))
        logging.info('loading dev data_cache_dir {}'.format(dev_examples_file))
        train_examples, dev_examples = utils_file.load(train_examples_file), utils_file.load(dev_examples_file)

        logging.info('train examples size is {}'.format(len(train_examples)))
        logging.info('dev examples size is {}'.format(len(dev_examples)))

    convert_examples_features = Feature(max_len=args.max_len, tokenizer=tokenizer)

    if debug:
        train_examples = train_examples[:2]
        dev_examples = dev_examples[:2]

    train_data_set = convert_examples_features(train_examples, data_type='train')
    dev_data_set = convert_examples_features(dev_examples, data_type='dev')
    train_data_loader = train_data_set.get_dataloader(args.train_batch_size, shuffle=False, pin_memory=args.pin_memory)
    dev_data_loader = dev_data_set.get_dataloader(args.train_batch_size)

    data_loaders = train_data_loader, dev_data_loader
    eval_examples = train_examples, dev_examples

    return eval_examples, data_loaders, tokenizer

