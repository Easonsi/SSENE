
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default="meituan", type=str, )
    parser.add_argument("--model", default="v1", type=str,)
    parser.add_argument("--exp_name", default="base", type=str, )
    
    # base
    parser.add_argument("--device_id", default=0, type=int, help="device id")
    parser.add_argument('--seed', type=int, default=2023, help="random seed for initialization")

    # file parameters
    parser.add_argument("--data_dir", default="./data", type=str) # , required=True)
    parser.add_argument("--log_dir", default="./logs", type=str, )
    parser.add_argument("--ckpt_dir", default="./saved", type=str, )

    # trainer parameters
    parser.add_argument('--train_mode', type=str, default="train")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--patience_stop', type=int, default=20, help='Patience for learning early stop')
    parser.add_argument('--eval_fuzzy', type=bool, default=False)

    # bert parameters
    parser.add_argument("--do_lower_case", action='store_true', )
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument('--pin_memory', type=bool, default=True)

    parser.add_argument("--base_model", default="Langboat/mengzi-t5-base-mt", type=str, help="model name")
    parser.add_argument("--dual_mode", default="consist", choices=["consist", "gen", "none"], type=str, help="model name")
    parser.add_argument("--fuse_mode", default="gate", choices=["gate", "gen", "none"], type=str, help="model name")
    
    parser.add_argument("--pooler", default="max", type=str, help="pooler: cls, mean, max")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument("--mlp_layers", type=int, default=1)

    # model parameters
    # parser.add_argument('--entity_emb_size', type=int, default=300)
    # parser.add_argument('--pos_limit', type=int, default=30)
    # parser.add_argument('--pos_dim', type=int, default=300)
    # parser.add_argument('--pos_size', type=int, default=62)

    # parser.add_argument('--bert_hidden_size', type=int, default=768)
    # parser.add_argument('--dropout', type=int, default=0.5)
    # parser.add_argument('--bidirectional', type=bool, default=True)


    args = parser.parse_args()
    args.data_dir = f'{args.data_dir}/{args.dataset}/'
    args.log_dir = f'{args.log_dir}/{args.dataset}/{args.model}/{args.exp_name}/'
    args.ckpt_dir = f'{args.ckpt_dir}/{args.dataset}/{args.model}/{args.exp_name}/'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.data_cache_dir = f'{args.data_dir}/cache_{args.base_model}_{args.max_len}/'
    os.makedirs(args.data_cache_dir, exist_ok=True)
    # args.f_ckp = f'{args.ckpt_dir}/ckp_{args.model}_{args.max_len}/' 
    return args






