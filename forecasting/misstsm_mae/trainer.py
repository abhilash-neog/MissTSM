import sys
sys.path.insert(1, './utils/.')

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

from model import MaskedAutoencoder, DecoderWithLinearHead 
from collections import OrderedDict
from functools import partial
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import lr_scheduler
from timm.models.vision_transformer import Block
from utils.util import MaskEmbed, MAEDataset, NativeScaler, get_1d_sincos_pos_embed, ActiveEmbed, FeatEmbed, adjust_learning_rate
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from tools import EarlyStopping, adjust_learning_rate, visual

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
                 n2one,
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
        
        if n2one==True:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, 1, bias=True).to(self.device)  # decoder to patch
        else:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, num_feats, bias=True).to(self.device)  # decoder to patch
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        enc_z = torch.rand((1, self.window_len + 1, self.num_feats, self.enc_embed_dim)).to(self.device) # +1 for the cls token
        self.pos_embed = self.pos_embed(enc_z)
        
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.window_len, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.window_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

class Trainer():
    
    def __init__(self, args, model, utils):
        self.args = args
        self.model = model
        self.batch_size = self.args['batch_size']
        self.accum_iter = self.args['accum_iter']
        # self.min_lr = self.args['min_lr']
        self.weight_decay = self.args['weight_decay']
        self.lr = self.args['lr']
        # self.blr = self.args['blr']
        # self.warmup_epochs = self.args['warmup_epochs']
        self.max_epochs = self.args['max_epochs']
        self.pct_start = self.args['pct_start']
        self.device = self.args['device']
        self.eval_freq = self.args['eval_freq']
        self.feature_wise_mse = self.args['feature_wise_mse']
        self.seq_len = self.args['seq_len']
        self.task_name = self.args['task_name']
        self.n2one_ft = self.args['n2one']
        self.utils = utils
        self.pred_len = self.args['pred_len']
        self.window_len = self.seq_len
        self.finetune_checkpoints_dir = args['finetune_checkpoints_dir']
        self.mpl = ModelPlugins(window_len=self.window_len, 
                                enc_embed_dim=self.model.embed_dim, 
                                dec_embed_dim=self.model.decoder_embed_dim,
                                num_feats=self.model.num_feats, 
                                task_name=self.task_name, 
                                n2one=self.n2one_ft, 
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
        
    def select_scaler_(self):
        loss_scaler = NativeScaler()
        return loss_scaler
    
    def get_data(self, X, split_flag):
        
        M = 1 - (1 * (torch.isnan(X)))
        M = M.float()
        
        X = torch.nan_to_num(X)
        
        '''
        Dataloader
        '''
        if split_flag=='test':
            dataset = MAEDataset(X, M)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size,
                drop_last=True
            )
        else:
            dataset = MAEDataset(X, M)
            dataloader = DataLoader(
                dataset, sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
            )
        return dataloader
    
    def val_one_epoch(self, dataloader, split, masked_penalize):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        
        self.model.eval()
        for iteration, (samples, masks) in enumerate(dataloader):
            samples = samples.to(self.device)
            masks = masks.to(self.device)

            with torch.cuda.amp.autocast():

                pred, mask, nask = self.model(samples, masks, self.mpl)

                if self.n2one_ft==True:
                    samples = samples[:, :, self.utils.target_index].unsqueeze(2)

                loss, masked_loss, unmasked_loss = self.model.forward_loss(data=samples, 
                                                                           pred=pred, 
                                                                           mask=mask, 
                                                                           nask=nask, 
                                                                           miss_idx=masks, 
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

            del samples
            del masks
        
        batch_loss /= len(dataloader)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)
        
        self.model.train()
        return batch_loss, masked_batch_loss, unmasked_batch_loss
        
    def train_one_epoch(self, dataloader, split, masked_penalize, optimizer, scheduler):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        
        optimizer.zero_grad()
        
        self.model.train()
        for iteration, (samples, masks) in enumerate(dataloader):
            samples = samples.to(self.device)
            masks = masks.to(self.device)

            with torch.cuda.amp.autocast():

                pred, mask, nask = self.model(samples, masks, self.mpl)
                
                if self.n2one_ft==True:
                    samples = samples[:, :, self.utils.target_index].unsqueeze(2)

                loss, masked_loss, unmasked_loss = self.model.forward_loss(data=samples, 
                                                                           pred=pred, 
                                                                           miss_idx=masks,
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
            scheduler.step()

            if (iteration + 1) % self.accum_iter == 0:
                optimizer.zero_grad()

            del samples
            del masks
        
        # scheduler.step()
            
        batch_loss /= len(dataloader)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)
        
        return batch_loss, masked_batch_loss, unmasked_batch_loss
        
    def pretrain(self, Xtrain=None, Xval=None, Xtest=None, masked_penalize=False):
        
        num_windows = self.args["num_windows"]
        num_samples = self.args["num_samples"]
        
        config = init_wandb(self.args, self.task_name)
        
        # Set missing Train
        M = 1 - (1 * (torch.isnan(Xtrain)))
        M = M.float()#.to(self.device)
        
        Xtrain = torch.nan_to_num(Xtrain)
        # Xtrain = Xtrain.to(self.device)
        
        # Set missing Val
        M_val = 1 - (1 * (torch.isnan(Xval)))
        M_val = M_val.float()#.to(self.device)
        
        Xval = torch.nan_to_num(Xval)
        # Xval = Xval.to(self.device)
        
        # Set missing Test
        M_test = 1 - (1 * (torch.isnan(Xtest)))
        M_test = M_test.float()#.to(self.device)
        
        Xtest = torch.nan_to_num(Xtest)
        # Xtest = Xtest.to(self.device)
        
        self.model.to(self.device)
        
        n_batches = int(math.ceil(Xtrain.shape[0] / self.batch_size))
        
        eff_batch_size = self.batch_size * self.accum_iter
        
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64

        '''
        Train dataloader
        '''
        train_dataset = MAEDataset(Xtrain, M)
        self.train_dataloader = DataLoader(
            train_dataset, sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size,
        )
        
        '''
        Val Dataloader
        '''
        val_dataset = MAEDataset(Xval, M_val)
        self.val_dataloader = DataLoader(
            val_dataset, sampler=RandomSampler(val_dataset),
            batch_size=self.batch_size,
        )
        
        '''
        Test Dataloader
        '''
        test_dataset = MAEDataset(Xtest, M_test)
        self.test_dataloader = DataLoader(
            test_dataset, sampler=RandomSampler(test_dataset),
            batch_size=self.batch_size,
            drop_last=True
        )
        
        losses = np.full(self.max_epochs, np.nan)
        val_mse = []
        train_mse = []
    
        torch.autograd.set_detect_anomaly(True)
        min_vali_loss=None
        
        optimizer = self.select_optimizer_()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                                     T_max=self.max_epochs)

        with trange(self.max_epochs) as tr:
            
            for it in tr:
                
                epoch_time = time.time()
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                
                train_loss, masked_train_loss, unmasked_train_loss = self.train_one_epoch(dataloader=self.train_dataloader, 
                                                                                          split='train', 
                                                                                          masked_penalize=masked_penalize,
                                                                                          optimizer=optimizer,
                                                                                          scheduler=model_scheduler)
                
                print("Epoch: {} cost time: {}".format(it + 1, time.time() - epoch_time))
                
                val_loss, masked_val_loss, unmasked_val_loss = self.val_one_epoch(dataloader=self.val_dataloader, 
                                                                                  split='val', 
                                                                                  masked_penalize=masked_penalize)
                
                losses[it] = train_loss
                               

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
                
                wandb.log(metrics)
                
                # checkpoint saving
                if not min_vali_loss or val_loss <= min_vali_loss:
                    if it == 0:
                        min_vali_loss = val_loss

                    print(
                        "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, val_loss, it))

                    min_vali_loss = val_loss
                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    path = os.path.join(self.args['pretrain_checkpoints_dir'], 'ckpt_best.pth')
                    torch.save(self.model, path)

                if (it + 1) % 10 == 0:
                    print("Saving model at epoch {}...".format(it + 1))

                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    path = os.path.join(self.args['pretrain_checkpoints_dir'], f'ckpt{it + 1}.pth')
                    torch.save(self.model, path)
            
            wandb.finish()
            
        return losses, self.model
    
    def finetune(self, Xtrain=None, Xval=None, Xtest=None, masked_penalize=False):
               
        num_windows = self.args["num_windows"]
        num_samples = self.args["num_samples"]
        
        masked_penalize=True
        print("Mask Penalize Has been set to True.")
        lookback = self.seq_len
        
        self.model.set_lookbackwindow(lookback, self.pred_len)
        
        self.model.set_masking_mode(masking_mode="continuous_masking")
        
        if self.args['freeze_encoder']=='True':
            self.freeze_encoder_model()
        
        config = init_wandb(self.args, self.task_name)
        
        self.model.to(self.device)
        
        n_batches = int(math.ceil(Xtrain.shape[0] / self.batch_size))
        
        eff_batch_size = self.batch_size * self.accum_iter
        
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64

        self.train_dataloader = self.get_data(Xtrain, split_flag='train')
        self.val_dataloader = self.get_data(Xval, split_flag='val')
        self.test_dataloader = self.get_data(Xtest, split_flag='test')
        
        losses = np.full(self.max_epochs, np.nan)
        val_mse = []
        train_mse = []
    
        torch.autograd.set_detect_anomaly(True)
        min_vali_loss=None
        
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=True)
        optimizer = self.select_optimizer_()
        train_steps = len(self.train_dataloader)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.pct_start,
                                            epochs=self.max_epochs,
                                            max_lr=self.lr)
        
        with trange(self.max_epochs) as tr:
            
            for it in tr:
                
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                
                batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        
                optimizer.zero_grad()

                self.model.train()
                epoch_time = time.time()
                
                for iteration, (samples, masks) in enumerate(self.train_dataloader):
                    
                    sample_X = copy.deepcopy(samples[:, :self.seq_len, :]).to(self.device)
                    sample_Y = copy.deepcopy(samples[:, -self.pred_len:, :]).to(self.device)
                    
                    mask_X = copy.deepcopy(masks[:, :self.seq_len, :]).to(self.device)
                    mask_Y = copy.deepcopy(masks[:, -self.pred_len:, :]).to(self.device)
                    
                    with torch.cuda.amp.autocast():

                        pred = self.model(sample_X, mask_X, self.mpl)
                        
                        if self.n2one_ft==True:
                            sample_Y = sample_Y[:, :, self.utils.target_index].unsqueeze(2)
                        
                        loss, masked_loss, unmasked_loss = self.model.forward_loss(data=sample_Y, 
                                                                                   pred=pred, 
                                                                                   miss_idx=mask_Y, 
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
                    scheduler.step()
                    
                    # we use a per iteration (instead of per epoch) lr scheduler
                    if (iteration + 1) % self.accum_iter == 0:
                        optimizer.zero_grad()
                
                print("Epoch: {} cost time: {}".format(it + 1, time.time() - epoch_time))
                batch_loss /= len(self.train_dataloader)
                
                val_mse = self.evaluate_forecast(model=self.model, dataloader=self.val_dataloader, args=self.args, train_or_val='val')['mse_dict']['MSE']
                test_mse = self.evaluate_forecast(model=self.model, dataloader=self.test_dataloader, args=self.args, train_or_val='test')['mse_dict']['MSE']
                
                if masked_loss is not None:
                    masked_batch_loss /= len(self.train_dataloader)
                    unmasked_batch_loss /= len(self.train_dataloader)
                
                losses[it] = batch_loss
                
                path = self.finetune_checkpoints_dir
                early_stopping(val_mse, self.model, path)
                # # if early_stopping.early_stop:
                # #     print("Early stopping")
                # #     break
                
                adjust_learning_rate(optimizer, scheduler, it + 1, self.args)
                

                metrics = {
                    "train_loss": batch_loss,
                    "val_mse": val_mse,
                    "test_mse": test_mse
                }                
                tr.set_postfix(metrics)
                
                wandb.log(metrics)
            
            return losses, self.model
    
    def wandb_summarize(self, val_eval_dict, train_or_test):
        
        mse=train_or_test+'_mse'
        mae=train_or_test+'_mae'
        
        for k,v in val_eval_dict.items():
            if k=='MSE':
                wandb.summary[mse] = v
            elif k=='MAE':
                wandb.summary[mae] = v
            else:
                wandb.summary[k] = v
        
    def predict(self, model, dataloader, args):
        model.eval() # setting model to eval mode
        
        self.batch_size = args['batch_size']
        self.accum_iter = args['accum_iter']
        self.weight_decay = args['weight_decay']
        self.max_epochs = args['max_epochs']
        self.device = args['device']
        
        n_batches = len(dataloader)

        samples_list = []
        preds_list = []
        masks_list = []
        nasks_list = []
        batch_loss = 0
        og_masks_list = []
        
        for it, (samples, masks) in tqdm(enumerate(dataloader)):
            
            samples = samples.to(self.device)
            masks = masks.to(self.device)
            
            with torch.cuda.amp.autocast():
                pred, mask, nask = model(samples, masks, self.mpl)
                
                if self.n2one_ft==True:
                    samples = samples[:, :, self.utils.target_index].unsqueeze(2)
                
            # samples_list.append(samples.transpose(1, 2).detach())
            samples_list.append(samples.detach())
            preds_list.append(pred.detach())
            masks_list.append(mask.detach())
            nasks_list.append(nask.detach())
            og_masks_list.append(masks.detach())
                    
            loss, _, _ = model.forward_loss(samples, pred, mask, nask, masks)

            loss_value = loss.item()
            batch_loss += loss_value
            del samples
            del masks
        
        samples_list = torch.cat(samples_list, dim=0)
        preds_list = torch.cat(preds_list, dim=0)
        masks_list = torch.cat(masks_list, dim=0)
        nasks_list = torch.cat(nasks_list, dim=0)
        og_masks_list = torch.cat(og_masks_list, dim=0)
        
        batch_loss = batch_loss/len(dataloader)
        
        metrics = {
            "batch_loss":batch_loss,
            "pred":preds_list,
            "samples":samples_list,
            "masks":masks_list,
            "nasks":nasks_list,
            "og_masks":og_masks_list
        }
        return metrics
    
    def evaluate_forecast(self, model, dataloader, args, train_or_val):
        model.eval() # setting model to eval mode
        
        self.batch_size = args['batch_size']
        self.accum_iter = args['accum_iter']
        self.weight_decay = args['weight_decay']
        
        self.max_epochs = args['max_epochs']
        self.device = args['device']
        
        n_batches = len(dataloader)

        samples_list = []
        preds_list = []
        batch_loss = 0
        og_masks_list = []
        
        for it, (samples, masks) in tqdm(enumerate(dataloader)):
            
            sample_X = copy.deepcopy(samples[:, :self.seq_len, :]).to(self.device)
            sample_Y = copy.deepcopy(samples[:, -self.pred_len:, :]).to(self.device)

            mask_X = copy.deepcopy(masks[:, :self.seq_len, :]).to(self.device)
            mask_Y = copy.deepcopy(masks[:, -self.pred_len:, :]).to(self.device)

            with torch.cuda.amp.autocast():
                pred = model(sample_X, mask_X, self.mpl)
                
                if self.n2one_ft==True:
                    sample_Y = sample_Y[:, :, self.utils.target_index].unsqueeze(2)
            
            loss, _, _ = model.forward_loss(data=sample_Y, 
                                                 pred=pred, 
                                                 miss_idx=mask_Y, 
                                                 masked_penalize=True)
            
            samples_list.append(sample_Y.detach())
            preds_list.append(pred.detach())
            og_masks_list.append(mask_Y.detach())

            loss_value = loss.item()
            batch_loss += loss_value
            del sample_X, sample_Y
            del mask_X, mask_Y
        
        val_X = torch.cat(samples_list, dim=0)
        predictions = torch.cat(preds_list, dim=0)
        og_masks = torch.cat(og_masks_list, dim=0)
        
        batch_loss = batch_loss/len(dataloader)
        
        twodmasks = og_masks
        
        MSE_dict = {}
        MAE_dict = {}
        
        with torch.cuda.amp.autocast():
            
            if self.feature_wise_mse=='True':
                '''
                MSE
                '''
                sqred_err = (predictions-val_X)**2
                sum_sqred_err = (sqred_err*twodmasks).sum((0,1)) # sum across batches and windows for each feature
                feature_wise_mse = (sum_sqred_err/twodmasks.sum((0,1)))
                MSE_dict = {train_or_val+"_"+self.utils.inp_cols[idx]:feature_wise_mse[idx].item() for idx in range(self.model.num_feats)}

                '''
                MAE
                '''
                abs_err = torch.abs(predictions - val_X)
                masked_abs_err = abs_err*twodmasks

                feature_wise_mae = masked_abs_err.sum((0, 1))/twodmasks.sum((0, 1))
                MAE_dict = {train_or_val+"_"+self.utils.inp_cols[idx]:feature_wise_mae[idx].item() for idx in range(self.model.num_feats)}
            
            MSE = ((((predictions-val_X)**2)*twodmasks).sum())/twodmasks.sum()
            MSE_dict["MSE"] = MSE.item()

            MAE = ((torch.abs(predictions-val_X)*twodmasks).sum())/twodmasks.sum()
            MAE_dict["MAE"] = MAE.item()

        return {'avg_loss':batch_loss, 'mse_dict':MSE_dict, 'mae_dict':MAE_dict, 'preds':predictions, 'gt': val_X, 'og_masks':og_masks}
        
        
    def test(self, model, test_X_):
        
        gt_test_dataloader = self.get_data(test_X_, split_flag='test')
        
        self.device = self.args['device']
        
        folder_path = self.args['output_path']
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        samples_list = []
        preds_list = []
        batch_loss = 0
        og_masks_list = []

        model.eval()
        with torch.no_grad():
            for it, (samples, masks) in tqdm(enumerate(self.test_dataloader)):
            
                sample_X = copy.deepcopy(samples[:, :self.seq_len, :]).to(self.device)
                sample_Y = copy.deepcopy(samples[:, -self.pred_len:, :]).to(self.device)

                mask_X = copy.deepcopy(masks[:, :self.seq_len, :]).to(self.device)
                mask_Y = copy.deepcopy(masks[:, -self.pred_len:, :]).to(self.device)

                with torch.cuda.amp.autocast():
                    pred = model(sample_X, mask_X, self.mpl)
                    
                    if self.n2one_ft==True:
                        sample_Y = sample_Y[:, :, self.utils.target_index].unsqueeze(2)
                
                preds_list.append(pred.detach())
                    
                del sample_X, sample_Y
                del mask_X, mask_Y
            
        for it, (samples, masks) in tqdm(enumerate(gt_test_dataloader)):
            
            sample_X = copy.deepcopy(samples[:, :self.seq_len, :]).to(self.device)
            sample_Y = copy.deepcopy(samples[:, -self.pred_len:, :]).to(self.device)

            mask_X = copy.deepcopy(masks[:, :self.seq_len, :]).to(self.device)
            mask_Y = copy.deepcopy(masks[:, -self.pred_len:, :]).to(self.device)
            
            if self.n2one_ft==True:
                sample_Y = sample_Y[:, :, self.utils.target_index].unsqueeze(2)
            
            samples_list.append(sample_Y.detach())
            og_masks_list.append(mask_Y.detach())
            
            if it % 100 == 0:
                inputX = sample_X.detach().cpu().numpy()
                gt = np.concatenate((inputX[0, :, -1], samples_list[it].cpu().numpy()[0, :, -1]), axis=0)
                pd = np.concatenate((inputX[0, :, -1], preds_list[it].cpu().numpy()[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(self.pred_len) + "_" + str(it) + '.pdf'))
            
            del sample_X, sample_Y
            del mask_X, mask_Y
        
        val_X = torch.cat(samples_list, dim=0)
        predictions = torch.cat(preds_list, dim=0)
        og_masks = torch.cat(og_masks_list, dim=0)
        
        twodmasks = og_masks
        
        MSE_dict = {}
        MAE_dict = {}
        
        with torch.cuda.amp.autocast():
            
            if self.feature_wise_mse=='True':
                '''
                MSE
                '''
                sqred_err = (predictions-val_X)**2
                sum_sqred_err = (sqred_err*twodmasks).sum((0,1)) # sum across batches and windows for each feature
                feature_wise_mse = (sum_sqred_err/twodmasks.sum((0,1)))
                MSE_dict = {"test_"+self.utils.inp_cols[idx]:feature_wise_mse[idx].item() for idx in range(self.model.num_feats)}

                '''
                MAE
                '''
                abs_err = torch.abs(predictions - val_X)
                masked_abs_err = abs_err*twodmasks

                feature_wise_mae = masked_abs_err.sum((0, 1))/twodmasks.sum((0, 1))
                MAE_dict = {"test_"+self.utils.inp_cols[idx]:feature_wise_mae[idx].item() for idx in range(self.model.num_feats)}
            
            MSE = ((((predictions-val_X)**2)*twodmasks).sum())/twodmasks.sum()
            MSE_dict["MSE"] = MSE.item()

            MAE = ((torch.abs(predictions-val_X)*twodmasks).sum())/twodmasks.sum()
            MAE_dict["MAE"] = MAE.item()
        
        mse = MSE_dict['MSE']
        mae = MAE_dict['MAE']
        
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.seq_len, self.pred_len, mse, mae))
        f = open(folder_path + "score_" + self.args['root_path'].split('/')[-1]+".txt", 'a')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.seq_len, self.pred_len, mse, mae))
        f.close()