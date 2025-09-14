import sys
# sys.path.insert(1, './utils/.')

import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import wandb
import sys
import os
import copy
import time
import matplotlib.pyplot as plt

from model import MaskedAutoencoder
from collections import OrderedDict
from functools import partial
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from utils import MaskEmbed, MAEDataset_PT, MAEDataset_FT, NativeScaler, get_1d_sincos_pos_embed, ActiveEmbed, FeatEmbed, adjust_learning_rate, cal_classification_metrics, get_class_weights
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from tools import EarlyStopping, adjust_learning_rate, transfer_weights

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from data_provider.clf_dataloader import data_generator

eps = 1e-6


def init_wandb(args, task_name):
    wandb.init(project=args['project_name'], 
               name="_".join([task_name, args['run_name']]), 
               config=args, 
               save_code=args['save_code'])
    config = wandb.config
    return config

class ModelPlugins():
    
    def __init__(self, 
                 window_len, 
                 enc_embed_dim,
                 dec_embed_dim,
                 task_name,
                 num_feats,
                 batch_size,
                 device):
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.window_len = window_len 
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.enc_embed_dim), requires_grad=False).to(self.device)
        self.pos_embed = PositionalEncoding2D(enc_embed_dim).to(self.device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.dec_embed_dim), requires_grad=False).to(self.device)
        # self.decoder_pos_embed = PositionalEncoding2D(dec_embed_dim).to(self.device)
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        enc_z = torch.rand((1, self.window_len + 1, self.num_feats, self.enc_embed_dim)).to(self.device) # +1 for the cls token
        self.pos_embed = self.pos_embed(enc_z)
        
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.window_len, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.window_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

class Trainer():
    
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(args.device)
        self.batch_size = self.args.batch_size
        self.accum_iter = self.args.accum_iter
        self.weight_decay = self.args.weight_decay
        self.lr = self.args.lr
        # self.train_epochs = self.args.train_epochs
        self.pct_start = self.args.pct_start
        self.device = self.args.device
        self.eval_freq = self.args.eval_freq
        self.seq_len = self.args.seq_len
        self.task_name = self.args.task_name
        self.window_len = self.seq_len
        self.finetune_checkpoints_dir = args.finetune_checkpoints_dir
        self.configs = self.model.data_config
        self.trial = args.trial
        
        # Load datasets
        self.sourcedata_path = os.path.join(args.clf_data_path, args.pretrain_dataset)
        self.targetdata_path = os.path.join(args.clf_data_path, args.target_dataset)
        
        self.mpl = ModelPlugins(window_len=self.window_len, 
                                enc_embed_dim=self.model.embed_dim, 
                                dec_embed_dim=self.model.decoder_embed_dim,
                                num_feats=self.model.num_feats, 
                                task_name=self.task_name, 
                                batch_size=self.batch_size,
                                device=self.device)
        
    def freeze_encoder_model(self):
        
        for param in self.model.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.model.mask_embed.parameters():
            param.requires_grad = False
        for param in self.model.mhca.parameters():
            param.requires_grad = False
                
        print(f"Encoder Blocks Frozen!")
        
    def select_optimizer_(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)#, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        return optimizer
        
    def select_scaler_(self):
        loss_scaler = NativeScaler()
        return loss_scaler
    
    def get_data(self, X, Y, split_flag):
        
        M = 1 - (1 * (torch.isnan(X)))
        M = M.float()
        
        X = torch.nan_to_num(X)
        
        '''
        Dataloader
        '''
        if split_flag=='test':
            dataset = MAEDataset_FT(X, Y, M)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size
            )
        else:
            dataset = MAEDataset_FT(X, Y, M)
            dataloader = DataLoader(
                dataset, sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
            )
        return dataloader
    
    def val_one_epoch(self, dataloader, split, masked_penalize):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        predictions = []
        gt = []
        self.model.eval()
        for iteration, (X, Y) in enumerate(dataloader):
            
            mask_original = 1 - (1 * (torch.isnan(X)))
            mask_original = mask_original.float().to(self.device)
            X = torch.nan_to_num(X).float().to(self.device)

            pred, mask, nask = self.model(X, mask_original, self.mpl)

            loss, masked_loss, unmasked_loss = self.model.forward_loss(data=X, 
                                                                       pred=pred, 
                                                                       mask=mask, 
                                                                       nask=nask, 
                                                                       miss_idx=mask_original, 
                                                                       masked_penalize=masked_penalize)

            if masked_loss is not None:
                masked_loss_value = masked_loss.item()
                unmasked_loss_value = unmasked_loss.item()

                masked_batch_loss += masked_loss_value
                unmasked_batch_loss += unmasked_loss_value

            loss_value = loss.item()
            batch_loss += loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            predictions.append(pred.detach().cpu().numpy())
            gt.append(X.detach().cpu().numpy())
        
        batch_loss /= len(dataloader)
        predictions = np.concatenate(predictions, axis=0)
        original = np.concatenate(gt, axis=0)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)
        
        self.model.train()
        return batch_loss, masked_batch_loss, unmasked_batch_loss, predictions, original
        
    def train_one_epoch(self, dataloader, split, masked_penalize, optimizer, scheduler):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        
        optimizer.zero_grad()
        
        self.model.train()
        for iteration, (X, Y) in enumerate(dataloader):
            
            mask_original = 1 - (1 * (torch.isnan(X)))
            mask_original = mask_original.float().to(self.device)
            X = torch.nan_to_num(X).float().to(self.device)
            # print(f"shape of X = {X.shape}")
            pred, mask, nask = self.model(X, mask_original, self.mpl)
            
            
            loss, masked_loss, unmasked_loss = self.model.forward_loss(data=X, 
                                                                       pred=pred, 
                                                                       miss_idx=mask_original,
                                                                       mask=mask, 
                                                                       nask=nask,
                                                                       masked_penalize=masked_penalize)
            if masked_loss is not None:
                masked_loss_value = masked_loss.item()
                unmasked_loss_value = unmasked_loss.item()

                masked_batch_loss += masked_loss_value
                unmasked_batch_loss += unmasked_loss_value

            loss_value = loss.item()
            batch_loss += loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= self.accum_iter
            
            loss.backward()
            optimizer.step()

            # we use a per iteration (instead of per epoch) lr scheduler
            if (iteration + 1) % self.accum_iter == 0:
                optimizer.zero_grad()

        scheduler.step()
        batch_loss /= len(dataloader)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)

        return batch_loss, masked_batch_loss, unmasked_batch_loss
        
    def pretrain(self):
        
        train_loader, vali_loader, _ = data_generator(self.sourcedata_path, 
                                                     self.targetdata_path, 
                                                     self.configs, 
                                                      self.trial,
                                                     training_mode='pre_train', 
                                                      
                                                     subset=False,
                                                     fraction=self.args.fraction)
        
        path = os.path.join(self.args.pretrain_checkpoints_dir, self.args.pretrain_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)
        
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                     T_max=self.args.pretrain_epochs)
        
        torch.autograd.set_detect_anomaly(True)
        min_vali_loss=None
        masked_penalize=False
        losses = []
        
        with trange(self.args.pretrain_epochs) as tr:
 
            # torch.cuda.empty_cache()
            
            for it in tr:
                
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                
                train_loss, masked_train_loss, unmasked_train_loss = self.train_one_epoch(dataloader=train_loader, 
                                                                                          split='train', 
                                                                                          optimizer=optimizer,
                                                                                          scheduler=model_scheduler,
                                                                                          masked_penalize=masked_penalize)
                val_loss, masked_val_loss, unmasked_val_loss, preds, gt = self.val_one_epoch(dataloader=vali_loader, 
                                                                                  split='val', 
                                                                                  masked_penalize=masked_penalize)
                
                if it==self.args.pretrain_epochs-1:
                    preds=np.squeeze(preds)
                    gt=np.squeeze(gt)

                    fig, axes = plt.subplots(10,1)
                    for idx, i in enumerate(np.random.choice(len(preds), 10, replace=False)):
                        axes[idx].plot(preds[i], label='predictions')
                        axes[idx].plot(gt[i], label='ground-truth')

                    plt.legend()
                    plt.savefig('./plots.pdf')

                
                losses.append(train_loss)
                

                if self.model.mask_ratio<1 and masked_penalize==False:
                    metrics = {
                        "train_loss": train_loss,
                        "train_masked_loss":masked_train_loss,
                        "train_unmasked_loss":unmasked_train_loss,
                        
                        "val_loss": val_loss,
                        "val_masked_loss":masked_val_loss,
                        "val_unmasked_loss":unmasked_val_loss,
                    }
                else:
                    metrics = {
                        "train_loss":train_loss,
                         "val_loss": val_loss,
                       }
                
                tr.set_postfix(metrics)
                
                # checkpoint saving
                if not min_vali_loss or val_loss <= min_vali_loss:
                    if it == 0:
                        min_vali_loss = val_loss

                    print(
                        "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, val_loss, it))

                    min_vali_loss = val_loss
                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    torch.save(self.model, os.path.join(path, "ckpt_best_"+self.args.fraction+".pth"))

                if (it + 1) % 10 == 0:
                    print("Saving model at epoch {}...".format(it + 1))
                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    torch.save(self.model, os.path.join(path, f'ckpt{it + 1}_' + self.args.fraction+'.pth'))
            
        return losses, self.model
        
    def finetune(self):
        
        train_loader, vali_loader, test_loader = data_generator(self.sourcedata_path,
                                                                self.targetdata_path, 
                                                                self.configs, 
                                                                 self.trial,
                                                                training_mode='fine_tune', 
                                                               
                                                                subset=False,
                                                               fraction=self.args.fraction)
        
        path = os.path.join(self.args.finetune_checkpoints_dir, self.args.target_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)
    
        torch.autograd.set_detect_anomaly(True)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, args=self.args)
        train_steps = len(train_loader)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.finetune_epochs,
                                            max_lr=self.args.lr)
        
        with trange(self.args.finetune_epochs) as tr:
            
            for it in tr:
                
                iter_count = 0
                train_loss = []

                total_loss = []
                total_acc = []
                total_auc = []
                total_prc = []

                outs = np.array([])
                trgs = np.array([])

                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                optimizer.zero_grad()

                self.model.train()
                for iteration, (X, labels) in enumerate(train_loader):
                    
                    mask_original = 1 - (1 * (torch.isnan(X)))
                    mask_original = mask_original.float().to(self.device)
                    X = torch.nan_to_num(X).float().to(self.device)
                    labels = labels.long().to(self.device)

                    predictions = self.model(X, mask_original, self.mpl)

                    loss = criterion(predictions, labels)
                    loss_value = loss.item()
                    
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                    loss /= self.accum_iter
                    
                    acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
                    onehot_label = F.one_hot(labels)
                    pred_numpy = predictions.detach().cpu().numpy()
                
                    try:
                        auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                    except:
                        auc_bs = 0.0

                    try:
                        prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                    except:
                        prc_bs = 0.0

                    total_acc.append(acc_bs)

                    if auc_bs != 0:
                        total_auc.append(auc_bs)
                    if prc_bs != 0:
                        total_prc.append(prc_bs)
                    total_loss.append(loss.item())
                
                    loss.backward()
                    optimizer.step()
                    
                    pred = predictions.max(1, keepdim=True)[1]
                    outs = np.append(outs, pred.cpu().numpy())
                    trgs = np.append(trgs, labels.data.cpu().numpy())
                
                    # we use a per iteration (instead of per epoch) lr scheduler
                    if (iteration + 1) % self.accum_iter == 0:
                        optimizer.zero_grad()
                
                labels_numpy = labels.detach().cpu().numpy()
                pred_numpy = np.argmax(pred_numpy, axis=1)
                F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

                total_loss = torch.tensor(total_loss).mean()  # average loss
                total_acc = torch.tensor(total_acc).mean()  # average acc
                total_auc = torch.tensor(total_auc).mean()  # average auc
                total_prc = torch.tensor(total_prc).mean()

                scheduler.step()
                
                vali_loss, _, _, _, _, vali_F1 = self.vali(vali_loader, criterion)
                test_loss, _, _, _, _, test_F1 = self.vali(test_loader, criterion)
            
                print(
                "Epoch: {0} | Train F1: {1:.7f} Vali F1: {2:.7f} Test F1: {3:.7f}".format(
                    it + 1, F1, vali_F1, test_F1))
                early_stopping(vali_loss, self.model, path)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break
                
                # adjust_learning_rate(optimizer, scheduler, it + 1, self.args)
                
        best_model_path = path + '/' + 'checkpoint_' + self.args.fraction + '.pth'
        self.model = torch.load(best_model_path, map_location='cpu')

        self.lr = optimizer.param_groups[0]['lr']

        return self.model
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        
        outs = np.array([])
        trgs = np.array([])
        print(f"Length of vali loader = {len(vali_loader)}")
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, labels) in enumerate(vali_loader):
                
                mask_original = 1 - (1 * (torch.isnan(batch_x)))
                mask_original = mask_original.float().to(self.device)
                batch_x = torch.nan_to_num(batch_x).float().to(self.device)
                labels = labels.long().to(self.device)

                # encoder
                outputs = self.model(batch_x, mask_original, self.mpl)

                loss = criterion(outputs, labels)
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels)
                pred_numpy = outputs.detach().cpu().numpy()
                
                try:
                    auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                except:
                    prc_bs = 0.0

                total_acc.append(acc_bs)

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_loss.append(loss.item())
                
                pred = outputs.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                
                # record
                total_loss.append(loss)
                
            labels_numpy = labels.detach().cpu().numpy()
            pred_numpy = np.argmax(pred_numpy, axis=1)
            F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

            total_loss = torch.tensor(total_loss).mean()  # average loss
            total_acc = torch.tensor(total_acc).mean()  # average acc
            total_auc = torch.tensor(total_auc).mean()  # average auc
            total_prc = torch.tensor(total_prc).mean()
                

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, total_acc, total_auc, total_prc, trgs, F1
    
    def test(self):
        _, _, test_loader = data_generator(self.sourcedata_path,
                                            self.targetdata_path, 
                                            self.configs, 
                                           self.trial,
                                            training_mode='fine_tune', 
                    
                                            subset=False,
                                            fraction=self.args.fraction)
        
        preds = []
        trues = []
        
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        total_precision, total_recall, total_f1 = [], [], []
        
        outs = np.array([])
        trgs = np.array([])

        self.model.eval().to(self.device)
        with torch.no_grad():
            
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            
            for i, (data, labels) in enumerate(test_loader):
                
                mask_original = 1 - (1 * (torch.isnan(data)))
                mask_original = mask_original.float().to(self.device)
                data = torch.nan_to_num(data).float().to(self.device)
                labels = labels.long().to(self.device)
                
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)

                # encoder
                outputs = self.model(data, mask_original, self.mpl)
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels)
                pred_numpy = outputs.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                
                try:
                    auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                except:
                    prc_bs = 0.0

                pred_numpy = np.argmax(pred_numpy, axis=1)
                precision = precision_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                recall = recall_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

                total_acc.append(acc_bs)

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(F1)

                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

                labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
                pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
            
        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]

        precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
        recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
        acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
        total_prc = torch.tensor(total_prc).mean()

        performance = {'acc': acc * 100, 
                       'precision':precision * 100, 
                       'recall':recall * 100, 
                       'F1':F1 * 100, 
                       'total_auc':total_auc * 100, 
                       'total_prc':total_prc * 100}
        return total_loss, total_acc, total_auc, total_prc, trgs, performance