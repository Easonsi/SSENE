
import logging
import os, sys
from warnings import simplefilter
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
simplefilter(action='ignore', category=FutureWarning)

from utils.init import set_logger, set_tb_writter, set_seed
from utils.parser import get_args
from utils.dataloader import Reader, Feature, build_dataset
from model.train import Trainer


model_dict = {
    "Langboat/mengzi-t5-base": 'mengzi-t5-base',
    "Langboat/mengzi-t5-base-mt": 'mengzi-t5-base-mt',
    "imxly/t5-pegasus": 't5-pegasus'
}

def main():
    args = get_args()
    args.exp_name = f"{datetime.date.today()}_fuse={args.fuse_mode}_dual={args.dual_mode}"
    
    logger:logging.Logger = set_logger(log_file=f"{args.log_dir}/train.log")
    writer = set_tb_writter(logdir=f"{args.log_dir}/tb")

    set_seed(args.seed)
    
    logger.info(f"pid: {os.getpid()}")
    logger.info(f"args: {args.__dict__}")
    logger.info(f"dataet: {args.dataset}")
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Experiment name: {args.exp_name}")
    logger.info(f"Train mode: {args.train_mode}")
    
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    # model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")
    reader = Reader()
    eval_examples, data_loaders, tokenizer = build_dataset(args, reader, tokenizer, debug=False)
    trainer = Trainer(args, data_loaders, eval_examples, tokenizer=tokenizer, writer=writer)

    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == 'gen':
        pass
    elif args.train_mode == "eval":
        trainer.resume(args)
        trainer.eval_data_set("dev")
    elif args.train_mode == "predict":
        trainer.resume(args)
        trainer.predict_data_set("dev")
    elif args.train_mode == "resume":
        trainer.resume(args)
        trainer.show("dev") 


if __name__ == '__main__':
    main()
