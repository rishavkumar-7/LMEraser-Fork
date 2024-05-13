from __future__ import print_function

import os
import sys

import torch
import torch.distributed as dist


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from arguments import Arguments
import utils.logging as logging
from utils.seed import set_seed
from eraser import Eraser
from data_utils import loader as data_loader
from models import builder as model_builder
from launch import logging_train_setup
from evaluation_metrics import Metrics
# =================================================
import torch
from PIL import Image
from torch.utils.data import DataLoader
# =================================================


def load_dataset(args):
    """Load datasets.
    """
    set_seed(args.seed)
    return [
        data_loader.construct_train_loader(args, args.test_dataset),
        data_loader.construct_val_loader(args, args.test_dataset),
        data_loader.construct_test_loader(args, args.test_dataset),
    ]

def main():
    # load datasets for meta train or test
    minis_test = load_dataset(args)

    # load pretrained model
    model, cur_device = model_builder._construct_model(args)
    # dist.init_process_group(backend='nccl', init_method='tcp://', rank=0, world_size=1)
    # torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=local_rank)
    # torch.distributed.init_process_group(world_size=args.num_gpus, rank=local_rank)
    init_method="file:///media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/lmeraser/shared_directory_for_peers/shared_communication.txt"
    torch.distributed.init_process_group(init_method=init_method,world_size=1, rank=0)
    # logger infor rank and world size
    logger.info(f"Rank: {torch.distributed.get_rank()}, World Size: {torch.distributed.get_world_size()}")

    # initialize meta-learner
    eraser = Eraser(args, model)
    eraser.model.to(cur_device)

    if args.erasing_method == "lmeraser":
        prompter_path = None if args.checkpoint_dir == "" else os.path.join(BASE_DIR, args.checkpoint_dir)
        loss,acc=eraser.LMErasing_tuning(minis_test, prompter_path) #minis_test is a list of type <class 'torch.utils.data.dataloader.DataLoader'>
        eval_met_graph=Metrics(loss,acc)
        eval_met_graph.loss_acc_graph(loss_filename=args.loss_filename,acc_filename=args.acc_filename)
    
    elif args.erasing_method == "random_part_tuning":
        prompter_path = None if args.checkpoint_dir == "" else os.path.join(BASE_DIR, args.checkpoint_dir)
        eraser.random_part_tuning(minis_test, prompter_path)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # parse arguments
    args = Arguments().parser().parse_args()
    args.device_type = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.local_rank = local_rank

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=local_rank) # not supported for mps and multi-node environments

    # setup training env including loggers
    logging_train_setup(args)
    logger = logging.get_logger("lmeraser")

    # basic configuration
    set_seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")

    # main loop
    main()