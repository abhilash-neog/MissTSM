__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']           

import torch
from torch import nn
import math
import copy
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    
# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
    
# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class TFI(nn.Module):
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

class LinearEmbed(nn.Module):
    """
    Embed each feature
    """
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, embedding_dim), nn.LayerNorm(embedding_dim))
        

    def forward(self, x):
        embedded_features = self.embedding(x.unsqueeze(-1))
        return embedded_features
    
class MissTSM(nn.Module):
    
    """ 
    Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, q_dim=8,
                 k_dim=8,
                 v_dim=8, 
                 num_feats=8, 
                 num_heads=1, 
                 out_dim=None, 
                 embed="linear", 
                 mtsm_norm=False,
                 layernorm=True):
        
        '''
        depth: refers to the number of encoder transformer blocks
        decoder_depth: refers to decoder transformer blocks
        mlp_ratio: is w.r.t ViT Block -> number of hidden layers = mlp_ratio*inp_size
        
        '''
        super().__init__()
        
        # TODO: Query dimension should be greater than key, value dimension
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.var_query = nn.Parameter(torch.zeros(1, 1, self.q_dim), requires_grad=True)
        self.num_feats = num_feats
        self.mtsm_norm = mtsm_norm
        self.embed = embed

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = num_feats
            
        ## Do we really need Multi-head attention?
        ## Grouped query-attention similar to llama3
        
        self.mhca = nn.MultiheadAttention(embed_dim=self.q_dim, num_heads=self.num_heads, batch_first=True)

        # self.mask_embed = LinearEmbed(embedding_dim=self.embed_dim)
        if self.embed=="linear":
            self.mask_embed = LinearEmbed(embedding_dim=self.q_dim)
        else:
            self.mask_embed = TFI(input_dim=self.num_feats, embedding_dim=self.q_dim)

        self.pos_embed = PositionalEncoding2D(self.q_dim)
        self.projection = nn.Linear(self.q_dim, self.out_dim)
        
        if layernorm:
            self.layernorm = nn.LayerNorm(self.q_dim)
        else:
            self.layernorm = None
        
    def RevIN(self, x, m):
        '''
        Perform Reversible instance normalization
        '''
        means = torch.sum(x, dim=1) / torch.sum(m == 1, dim=1)
        means = means.unsqueeze(1)
        x = x - means
        
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(m == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1)
        x /= stdev

        return x, means, stdev

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        
        m_ = copy.deepcopy(m.reshape(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward(self, x, m):
        
        # perform rev instance norm
        if self.mtsm_norm:
            x, means, std = self.RevIN(x, m)
        else:
            means, std = None, None
        
        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed(x)
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
            
        # perform cross-attention
        x = self.cross_attention(x, m)
            
        # linear projection
        x = self.projection(x)
        
        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        
        return x
    
class MissTSMSkip(nn.Module):
    
    """ 
    MissTSM with skip connections
    """
    
    def __init__(self, q_dim=8,
                 k_dim=8,
                 v_dim=8, 
                 num_feats=8, 
                 num_heads=1, 
                 out_dim=None, 
                 norm=False, 
                 embed="linear", 
                 mtsm_norm=False,
                 layernorm=True):
        
        '''
        depth: refers to the number of encoder transformer blocks
        decoder_depth: refers to decoder transformer blocks
        mlp_ratio: is w.r.t ViT Block -> number of hidden layers = mlp_ratio*inp_size
        
        '''
        super().__init__()
        
        # TODO: Query dimension should be greater than key, value dimension
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.var_query = nn.Parameter(torch.zeros(1, 1, self.q_dim), requires_grad=True)
        self.num_feats = num_feats
        self.norm = norm
        self.mtsm_norm = mtsm_norm
        self.embed = embed
        
        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = num_feats
            
        ## Do we really need Multi-head attention?
        ## Grouped query-attention similar to llama3
        
        self.mhca = nn.MultiheadAttention(embed_dim=self.q_dim, num_heads=self.num_heads, batch_first=True)

        # self.mask_embed = TFI(input_dim=self.num_feats, embedding_dim=self.embed_dim)
        if self.embed=="linear":
            self.mask_embed = LinearEmbed(embedding_dim=self.q_dim)
        else:
            self.mask_embed = TFI(input_dim=self.num_feats, embedding_dim=self.q_dim)
        
        self.pos_embed = PositionalEncoding2D(self.q_dim)
        self.projection = nn.Linear(self.q_dim, self.out_dim)
        
        if layernorm:
            self.layernorm = nn.LayerNorm(self.q_dim)
        else:
            self.layernorm = None
        
    def RevIN(self, x, m):
        '''
        Perform Reversible instance normalization
        '''
        means = torch.sum(x, dim=1) / torch.sum(m == 1, dim=1)
        means = means.unsqueeze(1)
        x = x - means
        
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(m == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1)
        x = x / stdev

        return x, means, stdev

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        
        m_ = copy.deepcopy(m.reshape(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward(self, x, m):
        
        # perform rev instance norm
        if self.mtsm_norm:
            # print(f"Applying RevIN to MissTSM")
            x, means, std = self.RevIN(x, m)
        else:
            means, std = None, None
        
        x_inp = x

        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed(x)

        # print(f"shape after positional embedding = {x.shape}")
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
            
        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # linear projection
        x = self.projection(x)
        
        x = m*x_inp + (1-m)*x

        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x