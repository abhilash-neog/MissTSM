import sys
sys.path.insert(1, './utils/')
sys.path.append('../')

import warnings
import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime
import timefeatures

from trainer import Trainer
from model import MaskedAutoencoder
from utils.util import Utils
from tools import transfer_weights
from functools import partial
from data_handler import DataHandler

warnings.filterwarnings('ignore')

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='benchmarktesting')

parser.add_argument('--task_name', type=str, required=True, default='pretrain', choices=['pretrain', 'finetune'], help='task name, options:[pretrain, finetune]')
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--dataset', type=str, required=True, default='ETT', help='dataset type')
parser.add_argument('--root_path', type=str, default='./forecasting_datasets/', help='root path of the data, code and model files')
parser.add_argument('--source_filename', type=str, default='ETTh1', help='name of the data file')
parser.add_argument('--gt_root_path', type=str, default=None, help='path to ground-truth data')
parser.add_argument('--gt_source_filename', type=str, default=None, help='path to ground-truth filename')
parser.add_argument('--timeenc', type=int, default=2, choices=[0, 1, 2], help='0 indicates traditional time features, 1 indicates time-features , 2 indicates no time-feature creation')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_best.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest.pth', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--pretrain_run_name', type=str, default='ett_pretrain_initial', help='run name in wandb')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')

parser.add_argument('--seq_len', type=int, default=336, help='window for pretraining and fine-tuning')
parser.add_argument('--label_len', type=int, default=48, help='window for pretraining and fine-tuning')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
parser.add_argument('--num_samples', type=int, default=10, help='number of sample regions to plot for each feature during pretraining')
parser.add_argument('--num_windows', type=int, default=25, help='number of windows to generate merged plots')
parser.add_argument('--feature_wise_mse', type=str, default='True', help='whether to plot feature-wise mse')

parser.add_argument('--pred_len', type=int, default=96, help='past sequence length')
parser.add_argument('--freeze_encoder', type=str, default='True', help='whether to freeze encoder or not')
parser.add_argument('--n2one', type=bool, default=False, help='multivariate featurest to univariate target')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--output_path', type=str, default='', help='path where all output are stored')
parser.add_argument('--trial', type=int, default=0)

# model define
parser.add_argument('--encoder_embed_dim', type=int, default=64, help='encoder embedding dimension in the feature space')
parser.add_argument('--encoder_depth', type=int, default=2, help='number of encoder blocks')
parser.add_argument('--encoder_num_heads', type=int, default=4, help='number of encoder multi-attention heads')
parser.add_argument('--decoder_depth', type=int, default=2, help='number of decoder blocks')
parser.add_argument('--decoder_num_heads', type=int, default=4, help='number of decoder multi-attention heads')
parser.add_argument('--decoder_embed_dim', type=int, default=32, help='decoder embedding dimension in the feature space')
parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio for vision transformer')

# training 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--accum_iter', type=int, default=1, help='accumulation iteration for gradient accumulation')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fc_dropout')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')

# weights and biases
parser.add_argument('--project_name', type=str, default='ett', help='project name in wandb')
parser.add_argument('--run_name', type=str, required=True, default='mae_pretraining_run', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')

args = parser.parse_args()

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

'''
update run name
'''
base_run_name = args.run_name
args.run_name = "{}_{}_{}".format(args.run_name, str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))


'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

    
'''
read and process data
'''
dh = DataHandler(args)
train_X, val_X, test_X, utils = dh.handle()

if args.task_name=='pretrain':
    
    model = MaskedAutoencoder(utils, args, num_feats=len(dh.handler.features_col))
    
    trainer = Trainer(args=vars(args), model=model, utils=utils)

    args.pretrain_checkpoints_dir = os.path.join(args.pretrain_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.pretrain_checkpoints_dir):
        os.makedirs(args.pretrain_checkpoints_dir)
    
    history, model = trainer.pretrain(train_X, val_X, test_X)
    
    model_path = os.path.join(args.pretrain_checkpoints_dir, args.ckpt_name)
    
    torch.save(model, model_path)
        
elif args.task_name=='finetune':
    
    args.finetune_checkpoints_dir = os.path.join(args.finetune_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)
    
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    model = MaskedAutoencoder(utils, args, num_feats=len(dh.handler.features_col))
    
    '''
    Training phase
    '''
    print(f"load_model_path = {load_model_path}")
    
    if os.path.exists(load_model_path):
        print(f"Transferring weights from pretrained model")
        model = transfer_weights(load_model_path, model, device=args.device)
    else:
        print(f"No pretrained weights found")
    
    trainer = Trainer(args=vars(args), model=model, utils=utils)
    
    history, model = trainer.finetune(train_X, val_X, test_X)
    
    save_model_path = os.path.join(args.finetune_checkpoints_dir, args.ckpt_name)
    torch.save(model, save_model_path) # saves the final model; may not be the best model
    
    '''
    Testing phase
    '''
    _, _, test_X_, _ = dh.handle(gt=True)
    
    best_model_path = os.path.join(args.finetune_checkpoints_dir, 'checkpoint.pth')
    
    print(f"Loading best fine-tuned model for testing ...")
    
    ft_model = torch.load(best_model_path, map_location='cpu').to(args.device)
    
    trainer.test(ft_model, test_X_)
    
print(f"Done with model {args.task_name} ")