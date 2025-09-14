import numpy as np
import torch, os
import pandas as pd
import math
import argparse
import wandb
import matplotlib.pyplot as plt
import pickle

from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, in_channel=14, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.proj = nn.Conv1d(in_channel, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class FeatEmbed(nn.Module):
    """
    Embed each feature
    """
    def __init__(self, input_dim=8, embedding_dim=8, norm_layer=None):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            for _ in range(input_dim)
        ])

    def forward(self, x):
        embedded_features = [emb_layer(x[:, :, i].unsqueeze(-1)) for i, emb_layer in enumerate(self.embeddings)]
        embedded_features = torch.stack(embedded_features, dim=2)
        return embedded_features

class ActiveEmbed(nn.Module):
    """ 
    record to mask embedding
    """
    def __init__(self, embed_dim=64, norm_layer=None):
        
        super().__init__()
        # self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):
        self.X = X
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]


class Utils:
    
    def __init__(self, inp_cols, date_col, args, stride=1):
        
        self.inp_cols = inp_cols
        self.date_col = date_col
        self.seq_len = args.seq_len
        self.stride = stride
        self.y_mean = None
        self.y_std = None
        self.device = args.device
        self.windowed_dataset_path = ""
        self.task_name = args.task_name
        self.n2one = args.n2one
        self.pred_len = args.pred_len

        if args.task_name=='finetune':
            self.pre_train_window = self.seq_len + self.pred_len
        else:
            self.pre_train_window = self.seq_len
        
        self.target_index = -1
    
    def load_pickle(self, path):
        
        with open(path, 'rb') as pickle_file:
            arr = pickle.load(pickle_file)
        
        return arr
        
    def split_data(self, df, ratios):
        '''
        For ETT we follow 6:2:2 ratio, and for other datasets, it is usually. 7:1:1
        '''
        total_rows = len(df)
        train_ratio = ratios['train']
        val_ratio = ratios['val']
        test_ratio = ratios['test']
        
        train_end_pt = int(train_ratio*total_rows)
        train_df = df[:train_end_pt].reset_index(drop='true')

        val_end_pt = train_end_pt + int(val_ratio*total_rows)
        val_df = df[train_end_pt:val_end_pt].reset_index(drop='true')

        test_df = df[val_end_pt:].reset_index(drop='true')
        
        return train_df, val_df, test_df
        
    
    def normalize_tensor(self, tensor, use_stat=False):
        
        eps = 1e-5 # epsilon for zero std
        
        '''
        use this when working on masked data
        '''
        if not use_stat:
            self.feat_mean = tensor.nanmean(dim=(0, 1))[None, None, :]
            mask = torch.isnan(tensor)
            filtered_data = tensor.clone()
            filtered_data[mask] = 0
            
            rev_mask = 1-(mask*1)
            
            sqred_values = rev_mask*((filtered_data - self.feat_mean)**2)
            sqred_sum = sqred_values.sum(dim=(0, 1))
            variance = sqred_sum/torch.sum(rev_mask, dim=(0, 1))
            
            self.feat_std = torch.sqrt(variance)[None, None, :]   
        
        tensor[:, :, :len(self.inp_cols)] = (tensor[:, :, :len(self.inp_cols)]-self.feat_mean)/(self.feat_std+eps)
        return tensor
        
    def normalize_pd(self, df, use_stat=False):
        '''
        Normalize data
        '''
        if use_stat:
            df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            return df
                
        # compute mean and std of target variable - to be used for unnormalizing
        self.feat_std = df[self.inp_cols].std(skipna=True)
        self.feat_mean = df[self.inp_cols].mean(skipna=True)
        
        df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            
        return df
        
    
    def numpy_to_torch(self, Xtrain, Ytrain, Xtest, Ytest):

        X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
        Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

        X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
        Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

        return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
    
    def perform_windowing(self, df, path, name, split='train'):
        
        filename=name + '_' + split + '_' + str(self.pre_train_window) +'.pkl'
        save_path=os.path.join(path, filename)
        
        if os.path.exists(save_path):
            print("Window dataset already exists")
            X = self.load_pickle(save_path)
            return X
        
        else:
            L = df.shape[0]
            num_samples = (L - self.pre_train_window) // self.stride + 1
            
            X = []
            
            for ii in tqdm(np.arange(num_samples)):
                start_x = self.stride * ii
                end_x = start_x + self.pre_train_window

                subset_df = df.iloc[start_x:end_x, :].copy(deep=True)
                
                X.append(np.expand_dims(subset_df, axis=0))

            X = np.concatenate(X, axis=0)
            
            return X