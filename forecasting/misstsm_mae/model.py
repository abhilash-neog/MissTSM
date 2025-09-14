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

from functools import partial
from torch.optim import lr_scheduler
from timm.models.vision_transformer import Block
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from utils.util import MaskEmbed, MAEDataset, NativeScaler, get_1d_sincos_pos_embed, ActiveEmbed, FeatEmbed, adjust_learning_rate

class DecoderWithLinearHead(nn.Module):
    
    def __init__(self, args, num_feats, cls_token, norm_layer=nn.LayerNorm):
        super().__init__()
        self.encoder_embed_dim = args.encoder_embed_dim
        self.decoder_embed_dim = args.decoder_embed_dim
        self.decoder_num_heads = args.decoder_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.norm_layer = norm_layer
        self.decoder_depth = args.decoder_depth
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_feats = num_feats
        self.cls_token = cls_token
        
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
        
        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.decoder_depth)])

        self.decoder_norm = self.norm_layer(self.decoder_embed_dim)
        
        self.decoder_inp = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.num_feats, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, means, std):  # [bs x seq_len x enc_embed_dim]
        
        x = x.permute(0, 2, 1) # [bs x enc_embed_dim x seq_len]
        x = self.decoder_inp(x) # [bs x enc_embed_dim x pred_len]
        x = x.permute(0, 2, 1) # [bs x pred_len x enc_embed_dim]
        
        x = self.decoder_embed(x)
        
        # append cls token
        cls_token = self.cls_token
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1, -1)
        
        cls_tokens = cls_tokens[:, :, 0, :]
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        # remove cls token
        x = x[:, 1:, :]
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        
        x = self.dropout(x)
        
        x = x * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return x

class FlattenHead(nn.Module):
    
    def __init__(self, args, num_feats):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_feats = num_feats
        self.encoder_embed_dim = args.encoder_embed_dim
        
        self.decoder_inp = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.decoder_pred = nn.Linear(self.encoder_embed_dim, self.num_feats, bias=True)
        self.dropout = nn.Dropout(args.fc_dropout)

    def forward(self, x, means, std):  # [bs x seq_len x d_model]
        
        x = x.permute(0, 2, 1) # [bs x enc_embed_dim x seq_len]
        x = self.decoder_inp(x) # [bs x enc_embed_dim x pred_len]
        x = x.permute(0, 2, 1) # [bs x pred_len x enc_embed_dim]
        
        x = self.decoder_pred(x) # [bs x pred_len x num_feats]
        
        x = self.dropout(x)
        
        x = x * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return x

class MaskedAutoencoder(nn.Module):
    
    """ 
    Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self,
                 utils,
                 args,
                 num_feats,
                 norm_layer=nn.LayerNorm, 
                 norm_field_loss=False,
                 encode_func='linear'):
        
        '''
        depth: refers to the number of encoder transformer blocks
        decoder_depth: refers to decoder transformer blocks
        mlp_ratio: is w.r.t ViT Block -> number of hidden layers = mlp_ratio*inp_size
        
        '''
        super().__init__()
        
        self.utils = utils
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # --------------------------------------------------------------------------
        self.embed_dim = args.encoder_embed_dim
        self.depth = args.encoder_depth
        self.num_heads= args.encoder_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.decoder_embed_dim = args.decoder_embed_dim
        self.decoder_num_heads = args.decoder_num_heads
        self.decoder_depth = args.decoder_depth
        self.mask_ratio = args.mask_ratio
        self.dropout = args.dropout
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        
        self.num_feats = num_feats
        self.norm_layer = norm_layer
        self.encode_func = encode_func
        self.norm_field_loss = norm_field_loss
        
        self.var_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        self.mhca = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

        self.mask_embed = FeatEmbed(input_dim=self.num_feats,
                                    embedding_dim=self.embed_dim,
                                    norm_layer=self.norm_layer)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.depth)])
        
        self.norm = self.norm_layer(self.embed_dim)

        if self.task_name=='pretrain':
            # --------------------------------------------------------------------------
            # MAE decoder specifics
            # --------------------------------------------------------------------------
            self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

            self.decoder_blocks = nn.ModuleList([
                Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
                for i in range(self.decoder_depth)])

            self.decoder_norm = self.norm_layer(self.decoder_embed_dim)

            # --------------------------------------------------------------------------

            self.norm_field_loss = self.norm_field_loss
            self.initialize_weights()
        
        elif self.task_name=='finetune':
            # ----------------
            # Decoder With a Linear Head
            # ----------------
            self.fh = FlattenHead(args, num_feats)
        
        self.set_masking_mode()
            
    def initialize_weights(self):
        
        # w = self.mask_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = [self.mask_embed.embeddings[i][0].weight.data for i in range(self.num_feats)]
        for i in range(self.num_feats):
            torch.nn.init.xavier_uniform_(w[i].view([w[i].shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_lookbackwindow(self, lookback, pred_len):
        # self.lookback_window = lookback
        # self.pred_len = pred_len
        print(f"Lookback window has been set to {self.seq_len}, forecast window has been set to {self.pred_len}")
    
    def set_masking_mode(self, masking_mode=None):
        if masking_mode is None:
            masking_mode = "random_masking"
        assert masking_mode in ["continuous_masking", "random_masking"]
        self.masking_mode = masking_mode
        print(f"Masking Mode has been set to {self.masking_mode}")
        
    def masking(self, x, m):
        if self.masking_mode=="random_masking":
            return self.random_masking(x, m)
        elif self.masking_mode=="continuous_masking":
            return self.continous_masking(x, m)
        else:
            print("Masking Error.")
    
    def continous_masking(self, x, m):
        N, L, D = x.shape  # batch, length, dim
        
        #uncomment this part when we infer
        len_keep = self.seq_len

        noise = torch.linspace(0, 1, L, device=x.device).repeat(N, 1)  # predictable noise
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask
        
        return x_masked, mask, nask, ids_restore
    
    def random_masking(self, x, m):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        
        #uncomment this part when we infer
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask
        
        return x_masked, mask, nask, ids_restore

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        
        m_ = copy.deepcopy(m.reshape(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward_encoder(self, x, m):
        
        '''
        rev-in
        '''
        means = torch.sum(x, dim=1) / torch.sum(m == 1, dim=1)
        means = means.unsqueeze(1)
        x = x - means
        
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(m == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1)
        x /= stdev
        
        # embed patches
        x = self.mask_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.mpl.pos_embed[:, 1:, :, :]
        
        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # masking: length -> length * mask_ratio
        if self.task_name=='pretrain':
            x, mask, nask, ids_restore = self.masking(x, m)
        
        # append cls token
        cls_token = self.cls_token + self.mpl.pos_embed[:, :1, :, :]
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1, -1)
        
        cls_tokens = cls_tokens[:, :, 0, :]
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x)
        
        if self.task_name=='pretrain':
            return x, mask, nask, ids_restore, means, stdev
        elif self.task_name=='finetune':
            return x, means, stdev

    def forward_decoder(self, x, ids_restore, means, std):
        # embed tokens
        
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.mpl.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        # x = torch.tanh(self.decoder_pred(x))/2 + 0.5
        x = self.mpl.decoder_pred(x)
        
        x = nn.Dropout(p=self.dropout)(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        x = x * (std[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return x
    

    def forward_loss(self, data, pred, miss_idx, mask=None, nask=None, masked_penalize=False):
        
        target = data
        if mask is not None:
            mask = mask.unsqueeze(-1) * torch.ones(1, pred.shape[2], device=mask.device)
            mask = mask*miss_idx
            nask = torch.ones([pred.shape[0], pred.shape[1], pred.shape[2]], device=mask.device) - mask
        else:
            mask = miss_idx #finetune
        
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps)**.5
        
        loss = (pred - target) ** 2
        
        masked_loss = None
        unmasked_loss = None
        
        if masked_penalize:
            loss = (loss * mask).sum() / mask.sum()
        else:
            if mask.sum()==0:
                loss = (loss * nask).sum() / nask.sum()
            else:
                masked_loss = (loss * mask).sum() / mask.sum()
                unmasked_loss = (loss * nask).sum() / nask.sum() 
                loss = masked_loss + unmasked_loss
                
        return loss, masked_loss, unmasked_loss


    def forward(self, data, miss_idx, mpl):
        
        self.mpl = mpl
        
        if self.task_name=='pretrain':
            latent, mask, nask, ids_restore, means, std = self.forward_encoder(data, miss_idx)
            pred = self.forward_decoder(latent, ids_restore, means, std)
            return pred, mask, nask
        
        elif self.task_name=='finetune':
            latent, means, std = self.forward_encoder(data, miss_idx)
            latent = latent[:, 1:, :]
            pred = self.fh(latent, means, std)
            return pred