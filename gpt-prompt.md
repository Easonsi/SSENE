
0-shot setting

给定一个句子, 抽取出句子中的否定三元组 <否定主语, 否定词, 否定内容>. 要求 (1) 否定词必须表达明确的语义上的否定, 例如 “不, 没有” 等; (2) 在抽取原句中的片段的同时, 尽量使得抽取的三元组可以连成一句语义完整的话. 

Given a sentence, extract the negation triple <subject, cue, scope>. The requirements are: (1) The negation cue must express explicit semantic negation, such as "不" (not) or "没有" (do not have). (2) While extracting fragments from the original sentence, try to ensure that the extracted triple can form a semantically complete sentence. 

5-shot setting

给定一个句子, 抽取出句子中的否定三元组 <否定主语, 否定词, 否定内容>. 要求 (1) 否定词必须表达明确的语义上的否定, 例如 “不, 没有” 等; (2) 在抽取原句中的片段的同时, 尽量使得抽取的三元组可以连成一句语义完整的话. 例如, 对于「楼盘之间还没建好的房子」这一短语, 我们需要抽取得到三元组 <房子, 没, 建好>. 
下面是5个例子, 你需要对后面的所给的句子进行抽取. 请直接以JSON格式输出三元组, 不需要加上句子.

Given a sentence, extract the negation triple <subject, cue, scope>. The requirements are: (1) The negation cue must express explicit semantic negation, such as "不" (not) or "没有" (do not have). (2) While extracting fragments from the original sentence, try to ensure that the extracted triple can form a semantically complete sentence. For example, for the phrase "楼盘之间还没建好的房子" (the houses that haven't been built between the real estate properties), we need to extract <房子, 没, 建好> (<house, not, built>).
Below are five examples, and you need to extract the negation triples for the given sentences. Please directly output the triples in JSON format, without including the original sentence.

10-shot setting

给定一个句子, 抽取出句子中的否定三元组 <否定主语, 否定词, 否定内容>. 要求 (1) 否定词必须表达明确的语义上的否定, 例如 “不, 没有” 等; (2) 在抽取原句中的片段的同时, 尽量使得抽取的三元组可以连成一句语义完整的话. 例如, 对于「楼盘之间还没建好的房子」这一短语, 我们需要抽取得到三元组 <房子, 没, 建好>. 
下面是10个例子, 你需要对后面的所给的句子进行抽取. 请直接以JSON格式输出三元组, 不需要加上句子.

Given a sentence, extract the negation triple <subject, cue, scope>. The requirements are: (1) The negation cue must express explicit semantic negation, such as "不" (not) or "没有" (do not have). (2) While extracting fragments from the original sentence, try to ensure that the extracted triple can form a semantically complete sentence. For example, for the phrase "楼盘之间还没建好的房子" (the houses that haven't been built between the real estate properties), we need to extract <房子, 没, 建好> (<house, not, built>).
Below are ten examples, and you need to extract the negation triples for the given sentences. Please directly output the triples in JSON format, without including the original sentence.


examples

"三点快四点的时候打电话质询的 问过有无儿童半价之类的 说是景区不售任何折扣票 下了订单换票的时候 看到景区写着四点半后50元购通票 好坑人" -> [["景区", "不", "售任何折扣票"]]
"游泳池水质不好，不清澈。" -> [["水质", "不", "好"], ["游泳池", "不", "清澈"]]
"景区门票太贵了，而且里面的景点也不怎么样，不值得去。", -> [["景点", "不", "怎么样"], ["景区", "不", "值得去"]]
"这个游戏厅的设备很老旧，而且场地也不大，真的不值得去。" -> [["场地", "不", "大"], ["游戏厅", "不", "值得去"]]
"四楼，胡安的画作，小鸟女人星星，这个是符号化的画作，比较难看懂" -> [["小鸟女人星星", "难", "看懂"]]
