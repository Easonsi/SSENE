# SSENE

The dataset `NegComment` is available in the folder `data/`, the dataset `SFU Review Corpus` can be downloaded from [here](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html). The used ChatGPT prompts can be found in `gpt-prompt.md`. 

## Environment

install the required packages in `requirements.txt`:

```text
pytorch==1.13.1
hanlp==2.1.0b48
transformers==4.26.1
sentencepiece==0.1.97
tensorboardX==2.6
numpy==1.23.5
pandas==1.5.3
tqdm
prettytable
```

## Run

```sh
python3 main.py
```

detailed parameters can be set in `utils/parser.py`

```sh
DID=0,1,2,3
HDIM=768
DUAL=consist
FUSE=gate
python -u main.py --model=v1 --device_id=$DID \
    --dataset=NegComment --exp_name=base \
    --train_batch_size=4 \
    --dual_mode=$DUAL --fuse_mode=$FUSE \
    --hidden_size=$HDIM
```

