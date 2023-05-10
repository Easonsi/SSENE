import os, re, sys, random, urllib.parse, json
from collections import defaultdict


# =================== JSON ===================
def SaveJson(obj, ofn, indent=2):
    with open(ofn, "w", encoding = "utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=indent)

def LoadJson(fn):
    with open(fn, encoding = "utf-8") as fin:
        return json.load(fin)

# =================== list, jsonl ===================
def LoadList(fn):
    with open(fn, encoding="utf-8") as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st

def LoadSet(fn):
    with open(fn, encoding="utf-8") as fin:
        st = set(ll for ll in fin.read().split('\n') if ll != "")
    return st

def LoadListg(fn):
    with open(fn, encoding="utf-8") as fin:
        for ll in fin:
            ll = ll.strip()
            if ll != '': yield ll

def SaveList(st, ofn):
    with open(ofn, "w", encoding = "utf-8") as fout:
        for k in st:
            fout.write(str(k) + "\n")

def LoadJsonlg(fn): return map(json.loads, LoadListg(fn))
def LoadJsonl(fn): return list(LoadJsonlg(fn))
def SaveJsonl(st, ofn): 
    return SaveList([json.dumps(x, ensure_ascii=False) for x in st], ofn)

# =================== CSV ===================
def WriteLine(fout, lst, sep='\t'):
    fout.write(sep.join([str(x) for x in lst]) + '\n')

def LoadCSV(fn, sep=", "):
    ret = []
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split(sep)
            ret.append(lln)
    return ret

def LoadCSVg(fn, sep=", "):
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split(sep)
            yield lln

def SaveCSV(csv, fn, sep=", "):
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            if ',' in sep:
                x = [str(i).replace(",", "，") for i in x]
            WriteLine(fout, x, sep)

# =================== dict ===================
def LoadDict(fn, func=str, sep="\t"):
    dict = {}
    with open(fn, encoding = "utf-8") as fin:
        for lv in (ll.split(sep, 1) for ll in fin.read().split('\n') if ll != ""):
            dict[lv[0]] = func(lv[1])
    return dict

def SaveDict(dict, ofn, sep="\t", output0=True):
    with open(ofn, "w", encoding = "utf-8") as fout:
        for k in dict.keys():
            if output0 or dict[k] != 0:
                fout.write(str(k) + sep + str(dict[k]) + "\n")



