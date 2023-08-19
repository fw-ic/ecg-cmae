# Author: Nabil Ibtehaz (https://github.com/nibtehaz)
# Insprired from https://github.com/facebookresearch/MAE

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from util.ecg_dataset import ECGPretrainDataset

#import models_mae
from MAEBank import MAEBank



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per model (effective batch size is batch_size * accum_iter * # gpus')        # default 64
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=2500, type=int,
                        help='images input size')
    parser.add_argument('--window_len', default=100, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--grad_accum_steps', type=int, default=1, metavar='LR',
                        help='grad_accum_steps')
                        

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/nabil/ecg_repr/physionet.org/files/challenge-2021/1.0.3/training/', type=str,
                        help='dataset path')
    
    parser.add_argument('--resume', type=str, help='resume training path')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume training path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device_count', default=4,
                        help='number of devices to use for training / testing')
    parser.add_argument('--seed', default=2, type=int)
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    parser.add_argument('--exp_name', default=' ',
                        help='name of experiment')

    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):

    args.output_dir = args.output_dir #+ '-' + args.exp_name

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=False)
        os.makedirs(os.path.join(args.output_dir,'logs'), exist_ok=False)
        os.makedirs(os.path.join(args.output_dir,'saved_models'), exist_ok=False)
        fp = open(os.path.join(args.output_dir , args.exp_name),'w')
        fp.close()
        
        sys.stdout = open(os.path.join(args.output_dir,'log.log'),'w')
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print('python3 '+ ' '.join(sys.argv))
    
    print('experiment: {}'.format(args.exp_name))

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
 
    devices = [torch.device(f'cuda:{i}') for i in range(args.device_count)]

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    #cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



    model_bnk = MAEBank(devices);


    if args.resume:
        print('resuming from ',args.resume)
        for i in range(len(model_bnk.maes)):
            chkpnt = torch.load(
                    f"experiments/{args.resume}/saved_models/mae_channel_{i+1}_best.pth",
                    map_location="cpu",
                )

            model_bnk.maes[i].load_state_dict(
                    chkpnt['model']
            )

            model_bnk.optimzers[i].load_state_dict(
                    chkpnt['optimizer']
            )

            model_bnk.schdulers[i].load_state_dict(
                    chkpnt['scheduler']
            )

 
    
    cur_best_loss = 100000

    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    
    # following timm: set wd as 0 for bias and norm layers


    
    print(f"Start training for {args.epochs} epochs")
    model_bnk.start_mp_daemons(args)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
