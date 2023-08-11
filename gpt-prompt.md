
## prompt for NegComment

Chinese version

```txt
给定一个句子, 抽取出句子中的否定三元组 <否定主语, 否定词, 否定内容>. 要求 (1) 否定词必须表达明确的语义上的否定, 例如 “不, 没有” 等; (2) 在抽取原句中的片段的同时, 尽量使得抽取的三元组可以连成一句语义完整的话. 例如, 对于「楼盘之间还没建好的房子」这一短语, 我们需要抽取得到三元组 <房子, 没, 建好>. 
下面是5个例子, 你需要对后面的所给的句子进行抽取. 请直接以JSON格式输出三元组, 不需要加上句子.
---
INPUT: "三点快四点的时候打电话质询的 问过有无儿童半价之类的 说是景区不售任何折扣票 下了订单换票的时候 看到景区写着四点半后50元购通票 好坑人"
OUTPUT: [{"subject": "景区", "cue" :"不", "scope": "售任何折扣票"}]
INPUT: "游泳池水质不好，不清澈。"
OUTPUT: [{"subject": "水质", "cue" :"不", "scope": "好"}, {"subject": "游泳池", "cue" :"不", "scope": "清澈"}]
INPUT: "景区门票太贵了，而且里面的景点也不怎么样，不值得去。"
OUTPUT: [{"subject": "景点", "cue" :"不", "scope": "怎么样"}, {"subject": "景区", "cue" :"不", "scope": "值得去"}]
INPUT: "这个游戏厅的设备很老旧，而且场地也不大，真的不值得去。"
OUTPUT: [{"subject": "场地", "cue" :"不", "scope": "大"}, {"subject": "游戏厅", "cue" :"不", "scope": "值得去"}]
INPUT: "四楼，胡安的画作，小鸟女人星星，这个是符号化的画作，比较难看懂"
OUTPUT: [{"subject": "小鸟女人星星", "cue" :"难", "scope": "看懂"}]
---
INPUT: [SENTENCE]
OUTPUT: 
```

English version

```txt
Given a sentence, extract the negation triple <subject, cue, scope>. The requirements are: (1) The negation cue must express explicit semantic negation, such as "不" (not) or "没有" (do not have). (2) While extracting fragments from the original sentence, try to ensure that the extracted triple can form a semantically complete sentence. For example, for the phrase "楼盘之间还没建好的房子" (the houses that haven't been built between the real estate properties), we need to extract <房子, 没, 建好> (<house, not, built>).
Below are 5 examples, and you need to extract the negation triples for the given sentences. Please directly OUTPUT the triples in JSON format, without including the original sentence.
---
INPUT: "三点快四点的时候打电话质询的 问过有无儿童半价之类的 说是景区不售任何折扣票 下了订单换票的时候 看到景区写着四点半后50元购通票 好坑人"
OUTPUT: [{"subject": "景区", "cue" :"不", "scope": "售任何折扣票"}]
INPUT: "游泳池水质不好，不清澈。"
OUTPUT: [{"subject": "水质", "cue" :"不", "scope": "好"}, {"subject": "游泳池", "cue" :"不", "scope": "清澈"}]
INPUT: "景区门票太贵了，而且里面的景点也不怎么样，不值得去。"
OUTPUT: [{"subject": "景点", "cue" :"不", "scope": "怎么样"}, {"subject": "景区", "cue" :"不", "scope": "值得去"}]
INPUT: "这个游戏厅的设备很老旧，而且场地也不大，真的不值得去。"
OUTPUT: [{"subject": "场地", "cue" :"不", "scope": "大"}, {"subject": "游戏厅", "cue" :"不", "scope": "值得去"}]
INPUT: "四楼，胡安的画作，小鸟女人星星，这个是符号化的画作，比较难看懂"
OUTPUT: [{"subject": "小鸟女人星星", "cue" :"难", "scope": "看懂"}]
---
INPUT: [SENTENCE]
OUTPUT: 
```


## prompt for SFU

```txt
Given a sentence, extract the negation cue and scope. Please return outputs in JSON format. {'cue': xxx, 'scope': xxx}
---
EXAMPLES
INPUT: "OTOH , when you buy a Honda or Acura , there are no options except color ."
OUTPUT: [{"cue": "no", "scope": "options"}]
INPUT: "These looks are not entirely deceiving ."
OUTPUT: [{"cue": "not", "scope": "entirely deceiving"}]
INPUT: "Naturally , I 'm not a big fan of static ."
OUTPUT: [{"cue": "not", "scope": "a big fan of static"}]
---
INPUT: Since the savings for buying via mail order are literally non-existant ( with sales tax and full retail prices still charged ) , there is literally no reason in the world not to buy retail .
OUTPUT: 
```

