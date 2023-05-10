

def bert_post_process(x):
    prediction=[]
    for i in x:
        out = i
        out = out.replace(' ','')
        out = out.replace('[SEP]','').replace('[PAD]','').replace('[CLS]','')
        # out=out.replace('<pad>','').replace('</s>','')
        prediction.append(out)
    return prediction

def spo2text(spo):
    return '+'.join(spo)
def spo_list2text(spos):
    return ';'.join([spo2text(spo) for spo in spos])

def split2spo_text(text):
    return text.strip().split(';')

