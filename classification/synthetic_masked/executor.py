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
from utils import Utils
from tools import transfer_weights
from functools import partial
from data_handler import DataHandler
from config_files.Epilepsy_Configs import Config as EConfigs
from config_files.SleepEEG_Configs import Config as SConfigs
from config_files.EMG_Configs import Config as EMConfigs
from config_files.Gesture_Configs import Config as GConfigs
from config_files.FDB_Configs import Config as FDBConfigs
warnings.filterwarnings('ignore')

'''
REMOVE NAME FROM THE CODE
'''

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='benchmarktesting')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', choices=['pretrain', 'finetune'], help='task name, options:[pretrain, finetune]')

# data loader
parser.add_argument('--clf_data_path', default='./data/', type=str, help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--pretrain_dataset', type=str, default='Epilepsy', help='name of the data file')
parser.add_argument('--target_dataset', type=str, default='Epilepsy', help='name of the data file')
parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune')

# model loader
parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_best.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest.pth', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')
parser.add_argument('--output_path', type=str, default='./outputs', help='path to output files')
parser.add_argument('--fraction', type=str, default='', help='fraction of missing data')
parser.add_argument('--trial', type=int, default=0)

# pretraining task
parser.add_argument('--seq_len', type=int, default=48, help='window for pretraining and fine-tuning')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')

# finetuning task
parser.add_argument('--freeze_encoder', type=str, default='True', help='whether to freeze encoder or not')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

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
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='dropout')
parser.add_argument('--finetune_epochs', type=int, default=100, help='dropout')

# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')


args = parser.parse_args()


num_feats={
    'Epilepsy':1,
    'SleepEEG':1,
    'Gesture': 3,
    'EMG':1,
    'FD-B':1
}

if args.pretrain_dataset == 'SleepEEG':
    args.seq_len = 178
    configs=SConfigs()
elif args.pretrain_dataset=='Epilepsy':
    args.seq_len = 178
    configs=EConfigs()
elif args.pretrain_dataset=='Gesture':
    args.seq_len = 178
    configs=GConfigs()
elif args.pretrain_dataset=='EMG':
    args.seq_len = 178
    configs=EMConfigs()
elif args.pretrain_dataset=='FD-B':
    args.seq_len = 178
    configs=FDBConfigs()
else:
    print("Wrong dataset")

    
'''
set num of target classes
'''
if args.target_dataset == 'SleepEEG':
    configs.num_classes_target=5
elif args.target_dataset=='Epilepsy':
    configs.num_classes_target=2
elif args.target_dataset=='Gesture':
    configs.num_classes_target=8
elif args.target_dataset=='EMG':
    configs.num_classes_target=3
elif args.target_dataset=='FD-B':
    configs.num_classes_target=3
else:
    print("Wrong dataset")

'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

if args.task_name=='pretrain':
    
    model = MaskedAutoencoder(args=args, data_config=configs, num_feats=num_feats[args.target_dataset])

    trainer = Trainer(args=args, model=model)
    
    if not os.path.exists(args.pretrain_checkpoints_dir):
        os.makedirs(args.pretrain_checkpoints_dir)
    
    history, model = trainer.pretrain()
    
    args.ckpt_name = args.ckpt_name[:-4] + '_' + args.fraction + '.pth'
    
    model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_dataset + "_v" + str(args.trial), args.ckpt_name)
    
    torch.save(model, model_path)
        
elif args.task_name=='finetune':
    
    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)
    
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_dataset + "_v" + str(args.trial), "ckpt_best_" + args.fraction+".pth")
    
    model = MaskedAutoencoder(args=args, data_config=configs, num_feats=num_feats[args.target_dataset]).to(args.device)
    
    model = transfer_weights(load_model_path, model, device=args.device)
    
    trainer = Trainer(args=args, model=model)
    model = trainer.finetune()
    
    save_model_path = os.path.join(args.finetune_checkpoints_dir, args.target_dataset +  "_v" + str(args.trial), "ckpt_latest_" +args.fraction+".pth")
    torch.save(model, save_model_path)
    
    total_loss, total_acc, total_auc, total_prc, trgs, performance = trainer.test()
    
    output_path = os.path.join(args.output_path, args.target_dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(output_path+'/metrics_' + args.fraction + '_v' + str(args.trial) +'.txt', 'w') as file:
        file.write(f"acc: {performance['acc']}\n")
        file.write(f"F1-score: {performance['F1']}\n")
        file.write(f"precision: {performance['precision']}\n")
        file.write(f"recall: {performance['recall']}\n")
    
print(f"Done with model {args.task_name} ")