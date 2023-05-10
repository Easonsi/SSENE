import os,sys
import logging

from tensorboardX import SummaryWriter

import random
import numpy as np
import torch

def set_logger(log_file) -> logging.Logger:
    """设置日志"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # 设置日志格式
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # 设置控制台日志
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # 设置文件日志
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

def set_tb_writter(logdir):
    # 若有的话, 删除之前的记录
    if os.path.exists(logdir):
        os.system('rm -rf {}'.format(logdir))
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    return writer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
