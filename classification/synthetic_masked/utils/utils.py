import numpy as np
import torch, os
import pandas as pd
import math
import argparse
import wandb
import matplotlib.pyplot as plt
import pickle

from sklearn import metrics
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def count_labels(dataloader):
    label_counts = {0: 0, 1: 0}  # Initialize count for each label

    for _, labels, _ in dataloader:
        for label in labels:
            label_counts[label.item()] += 1

    return label_counts

def get_class_weights(dataloader):    
    # Assuming you have a dataloader named 'train_dataloader'
    label_counts = count_labels(dataloader)
    total_samples = sum(label_counts.values())
    class_weights = [total_samples / count for count in label_counts.values()]
    return torch.tensor(class_weights)

def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )


def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_test, probas_pred=y_pred
    )
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds


def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds


def auc_to_recall(recalls, precisions, recall=0.01):
    precisions_mod = precisions.copy()
    ind = np.where(recalls < recall)[0][0] + 1
    precisions_mod[:ind] = 0
    area = metrics.auc(recalls, precisions_mod)
    return area

def cal_classification_metrics(probabilities, labels, pos_label=1, class_num=1):
    """
    pos_label: The label of the positive class.
    """
    if class_num == 1:
        class_predictions = (probabilities >= 0.5).astype(int)
    elif class_num == 2:
        class_predictions = np.argmax(probabilities, axis=1)
    else:
        assert "args.class_num>2, class need to be specified for precision_recall_fscore_support"
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, class_predictions, pos_label=pos_label, warn_for=()
    )
    precision, recall, f1 = precision[1], recall[1], f1[1]
    precisions, recalls, _ = metrics.precision_recall_curve(
        labels, probabilities[:, -1], pos_label=pos_label
    )
    acc_score = metrics.accuracy_score(labels, class_predictions)
    ROC_AUC, fprs, tprs, thresholds = auc_roc(probabilities[:, -1], labels)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        "classification_predictions": class_predictions,
        "acc_score": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precisions": precisions,
        "recalls": recalls,
        "fprs": fprs,
        "tprs": tprs,
        "ROC_AUC": ROC_AUC,
        "PR_AUC": PR_AUC,
    }
    return classification_metrics

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
        # print(f"x shape = {x.shape}")
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


class MAEDataset_PT(Dataset):

    def __init__(self, X, M):
        self.X = X
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]
    
class MAEDataset_FT(Dataset):

    def __init__(self, X, Y, M):
        self.X = X
        self.Y = Y
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.M[idx]


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
        # print(f"tensor[] self inp cols shape = {tensor[:, :, self.inp_cols].shape}")
        # if not use_stat:
        #     self.feat_mean = tensor[:, :, :len(self.inp_cols)].mean(dim=(0, 1))[None, None, :]
        #     self.feat_std = tensor[:, :, :len(self.inp_cols)].std(dim=(0, 1))[None, None, :]
        
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
        '''
        convert numpy array to PyTorch tensor
    
        : param Xtrain:               windowed training input data (# examples, input window size, # features); np.array
        : param Ytrain:               windowed training target data (# examples, output window size, # features); np.array
        : param Xtest:                windowed test input data (# examples, input window size, # features); np.array
        : param Ytest:                windowed test target data (# examples, output window size, # features); np.array
        : return X_train_torch, Y_train_torch,
        :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

        '''

        X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
        Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

        X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
        Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

        return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
    
    def perform_windowing(self, df, path, name, split='train'):
        '''
        create a windowed dataset
    
        : param y:                time series feature (array)
        : param input_window:     number of y samples to give model
        : param output_window:    number of future y samples to predict
        : param stide:            spacing between windows
        : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
        : return X, Y:            arrays with correct dimensions for LSTM
        :                         (i.e., [input/output window size # examples, # features])
        '''
        
        filename=name + '_' + split + '_' + str(self.pre_train_window) +'.pkl'
        save_path=os.path.join(path, filename)
        
        if os.path.exists(save_path):
            print("Window dataset already exists")
            X = self.load_pickle(save_path)
            return X
        
        else:
            L = df.shape[0]
            num_samples = (L - self.pre_train_window) // self.stride + 1

            # dfX = df[self.inp_cols]

            X = [] #np.zeros([num_samples, self.pre_train_window, self.num_features])
            # target_X = np.zeros([self.input_window, num_samples, self.num_out_features])
            # shuffled_inds = random.sample(range(num_samples),num_samples)
            
            for ii in tqdm(np.arange(num_samples)):
                start_x = self.stride * ii
                end_x = start_x + self.pre_train_window

                subset_df = df.iloc[start_x:end_x, :].copy(deep=True)
                
                X.append(np.expand_dims(subset_df, axis=0))
                
                # if X.shape[0]==0:
                #     X = np.expand_dims(subset_dfX, axis=0)
                # else:
                #     toAdd = np.expand_dims(subset_dfX, axis=0)
                #     X = np.append(X, toAdd, axis=0)

            X = np.concatenate(X, axis=0)
            
            # with open(save_path, 'wb') as pickle_file:
            #     pickle.dump(X, pickle_file)
            #     print(f"Pickled dataset {filename}")

            return X

    def plot_forecast(self, df, preds, og_masks, sample_idx, plt_idx, lookback_window, epoch, train_or_val, title_prefix):
        """
        This function plots t+plt_idx horizon plots
        """
        preds = preds.detach()
        df = df.detach()
        og_masks = og_masks.detach()
        
        sample_time_series = df[sample_idx] # GT : samples, L, feat
        predictions = preds[sample_idx]
        
        og_mask = og_masks[sample_idx]
        
        sample_time_series = sample_time_series*og_mask
        sample_time_series = torch.where(sample_time_series==0, torch.tensor(float('nan')).to(self.device), sample_time_series)

        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()

        # Create a figure
        num_feats = predictions.shape[2]
        num_samples = len(plt_idx)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(6*num_samples, num_feats*3))
        
        if self.n2one=="True":
            
            for i, sample_id in enumerate(plt_idx):      
                ax = axes[i]
                sample_id = int(sample_id)
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[0])
                ax.plot(dates, predictions[:, sample_id, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[:, sample_id, 0], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')

                subtitle = f'{feature_name}: Plot at t+{sample_id-lookback_window+1}'
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            
            for i, sample_id in enumerate(plt_idx):
                
                for idx in range(num_feats):
                    
                    ax = axes[idx, i]      
                    sample_id = int(sample_id)
                    feature_name = self.inp_cols[idx]
                    dates = np.arange(predictions.shape[0])
                    ax.plot(dates, predictions[:, sample_id, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                            color='green')
                    ax.plot(dates, sample_time_series[:, sample_id, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                             color='blue')

                    subtitle = f'{feature_name}: Plot at t+{sample_id-lookback_window+1}'
                    ax.set_title(subtitle)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Values')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.legend(loc="best")
            
        
        title = f'{title_prefix}: {train_or_val} t+N Forecasts at epoch: {epoch}'
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
        

    def plot_merged_context_windows(self, df, preds, og_masks, masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function plots merged context windows for random masking. 
        It assumes that stride for windowed dataset gen is 1 and data is not shuffled.
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        og_masks = og_masks.detach()
        
        # Extract the selected sample from the validation set and mask
        dates = range(len(sample_index)*self.pre_train_window)

        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]
        og_mask = og_masks[sample_index]

        ts = sample_time_series[0]
        ps = predictions[0]
        ms = mask[0]
        om = og_mask[0]
        
        for i in range(1, len(sample_index)):
            ts = torch.cat((ts, sample_time_series[i]), dim=0)
            ps = torch.cat((ps, predictions[i]), dim=0)
            ms = torch.cat((ms, mask[i]), dim=0)
            om = torch.cat((om, og_mask[i]), dim=0)
        
        # print(f"ts shape ={ts.shape} \nps shape = {ps.shape} \nms shape = {ms.shape}")
        
        # shape of ms = (875,)
        om = 1 - om
        masked_ts = ms.unsqueeze(-1) * torch.ones(1, ts.shape[1], device=ms.device)
        masked_ts = torch.logical_or(masked_ts, om)
        masked_ts = masked_ts*ts 
        # shape (875, 8)
        
#         print(f"ms shape = {ms.shape}")
#         print(f"masked_ts shape = {masked_ts.shape}")
                
        # unmasked_ts = (1-ms).unsqueeze(1)*ts

        masked_ts = torch.where(masked_ts==0, torch.tensor(float('nan')).to(self.device), masked_ts)
        # unmasked_ts = torch.where(unmasked_ts== 0, torch.tensor(float('nan')).to(device), unmasked_ts)

        masked_ts = masked_ts.cpu().numpy()
        # unmasked_ts = unmasked_ts.cpu().numpy()

        ts = ts.cpu().numpy()                          
        ps = ps.cpu().numpy()
        ms = ms.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        fig, axes = plt.subplots(num_feats, 1, figsize=(10, num_feats*3))
        for idx in range(num_feats):
            ax = axes[idx]
            
            # if self.task_name=="finetune":
            #     feature_name = self.inp_cols[self.chloro_index]
            # else:
            feature_name = self.inp_cols[idx]
            
            ax.plot(dates, ps[:, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                    color='green')
            ax.plot(dates, ts[:, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                     color='blue')
            ax.plot(dates, masked_ts[:, idx], label='Masked TS', marker='o', linestyle='-', markersize=1,
                     markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
            
            subtitle = '{}'.format(feature_name)
            ax.set_title(subtitle)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Values')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.legend(loc="best")
        
        title = '{}: {} Merged Context-Windows at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    
    def plot_context_window_grid_with_original_masks(self, df, preds, og_masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        og_masks = og_masks.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        og_mask = og_masks[sample_index]
        
        og_masked_ts = og_mask*sample_time_series
        og_masked_ts = torch.where(og_masked_ts==0, torch.tensor(float('nan')).to(self.device), og_masked_ts)
        
        masked_ts = og_masked_ts.cpu().numpy()
        og_masked_ts = og_masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        
        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))
        
        if self.n2one=="True":
            
            for sample_id in range(num_samples):
                # for idx in range(num_feats):
                # ax = axes[idx, sample_id]      
                ax = axes[sample_id]
                
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[1])
                plt.tight_layout()
                ax.plot(dates, predictions[sample_id, :, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, masked_ts[sample_id, :, 0], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                # ax.plot(dates, masked_ts[sample_id, :, 0], label='Original Masked TS', marker='o', linestyle='-', markersize=1,
                #          markerfacecolor='yellow', markeredgecolor='yellow', color='yellow', alpha=0.9)
                ax.axvline(x=self.seq_len, color='black', linestyle='-', linewidth=2)
                
                subtitle = '{}'.format(feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            for sample_id in range(num_samples):
                
                for idx in range(num_feats):
                    ax = axes[idx, sample_id]      

                    feature_name = self.inp_cols[idx]
                    dates = np.arange(predictions.shape[1])
                    ax.plot(dates, predictions[sample_id, :, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                            color='green')
                    ax.plot(dates, masked_ts[sample_id, :, idx], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                    # ax.plot(dates, masked_ts[sample_id, :, idx], label='Original Masked TS', marker='o', linestyle='-', markersize=1,
                    #          markerfacecolor='yellow', markeredgecolor='yellow', color='yellow', alpha=0.9)
                    
                    if self.task_name=='finetune' or self.task_name=='zeroshot':
                        plt.tight_layout()
                        ax.axvline(x=self.seq_len, color='black', linestyle='-', linewidth=2)
                        
                    subtitle = '{}'.format(feature_name)
                    ax.set_title(subtitle)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Values')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.legend(loc="best")
        
        title = '{}: {} Single Windows w/Original Masks at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_context_window_grid(self, df, preds, masks, og_masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        og_masks = og_masks.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]
        og_mask = og_masks[sample_index]
        
        og_mask = 1-og_mask
        mask = mask.unsqueeze(-1) * torch.ones(1, predictions.shape[2], device=mask.device)
        mask = torch.logical_or(mask, og_mask)
        masked_ts = mask*sample_time_series
        masked_ts = torch.where(masked_ts==0, torch.tensor(float('nan')).to(self.device), masked_ts)
        
        masked_ts = masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        mask = mask.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))

        if self.n2one=="True":
            
            for sample_id in range(num_samples):
                # for idx in range(num_feats):
                # ax = axes[idx, sample_id]      
                ax = axes[sample_id]
                
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[1])
                plt.tight_layout()
                ax.plot(dates, predictions[sample_id, :, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[sample_id, :, 0], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                ax.plot(dates, masked_ts[sample_id, :, 0], label='Masked TS', marker='o', linestyle='-', markersize=1,
                         markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
                
                ax.axvline(x=self.seq_len, color='black', linestyle='-', linewidth=2)
                
                subtitle = '{}'.format(feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            for sample_id in range(num_samples):
                for idx in range(num_feats):
                    ax = axes[idx, sample_id]      

                    feature_name = self.inp_cols[idx]
                    dates = np.arange(predictions.shape[1])
                    ax.plot(dates, predictions[sample_id, :, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                            color='green')
                    ax.plot(dates, sample_time_series[sample_id, :, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                             color='blue')
                    ax.plot(dates, masked_ts[sample_id, :, idx], label='Masked TS', marker='o', linestyle='-', markersize=1,
                             markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
                    
                    if self.task_name=='finetune' or self.task_name=='zeroshot':
                        plt.tight_layout()
                        ax.axvline(x=self.lookback_window, color='black', linestyle='-', linewidth=2)
                    
                    subtitle = '{}'.format(feature_name)
                    ax.set_title(subtitle)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Values')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.legend(loc="best")
        
        title = '{}: {} Individual Context-Windows at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()