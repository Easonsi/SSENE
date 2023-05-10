import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from easonsi import utils
from easonsi.util.leetcode import *


def process_pre_cnesp(idir, ofn):
    data = []
    for fn in os.listdir(idir):
        if not fn.endswith('.jsonl'):
            continue
        dtype = fn.split('.')[0]
        for l in utils.LoadJsonl(f"{idir}/{fn}"):
            s = l['sentence']
            label = []
            for c_idx, s_idx in l['labels']:
                label.append([''.join(s[c_idx[0]:c_idx[-1]+1]), ''.join(s[s_idx[0]:s_idx[-1]+1])])
            data.append({
                'type': dtype,
                'text': ''.join(s),
                'label': label
            })
    utils.SaveJsonl(data, ofn)


def process_syntax_tree(fn, ofn):
    import hanlp
    print(f"Loading hanlp models...")
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
    print(f"Loading hanlp models done.")
    processed = []
    print(f"Processing {fn}...")
    for d in tqdm(utils.LoadJsonl(fn)):
        tokens = tok(d['text'])
        tree = dep(tokens, conll=False)
        res = d.copy()
        res.update({
            'token': tokens,
            'head': [i[0] for i in tree],
            'deprel': [i[1] for i in tree], 
        })
        processed.append(res)
    print(f"Done, saved to {ofn}")
    utils.SaveJsonl(processed, ofn)


def split_data(fn, ddir):
    data = utils.LoadJsonl(fn)
    
    import random
    random.seed(1)
    random.shuffle(data)
    datasize = len(data)
    sampled = data[:datasize]
    p1, p2 = int(datasize*.8), int(datasize*.9)
    for dname, d in zip(
        ['train', 'dev', 'test'],
        [sampled[:p1], sampled[p1:p2], sampled[p2:]]
    ):
        utils.SaveJsonl(d, f"{ddir}/{dname}.json")


def get_syntax_tree(text):
    import hanlp
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
    tokens = tok(text)
    tree = dep(tokens, conll=False)
    res = {
        'text': text,
        'token': tokens,
        'head': [i[0] for i in tree],
        'deprel': [i[1] for i in tree], 
    }
    return res

if __name__=="__main__":
    ddir = './data/meituan'
    # process_pre_meituan(f'{ddir}/data-triple/triple.xlsx', f'{ddir}/all_raw.jsonl')
    process_syntax_tree(f'{ddir}/all_raw.jsonl', f'{ddir}/all_syntax.jsonl')
    split_data(f"{ddir}/all_syntax.jsonl", ddir)

