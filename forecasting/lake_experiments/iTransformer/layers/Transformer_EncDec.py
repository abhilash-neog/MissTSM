import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

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

class iMissTSM(nn.Module):
    
    """ 
    Masked Autoencoder with Transformer backbone
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
                 layernorm=True,
                 seq_len=336):
        
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
        self.seq_len = seq_len

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = seq_len
            
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
        var_query = self.var_query.repeat_interleave(batch_size*num_feat, dim=0)
        
        x = x.reshape(-1, window_size, d)
        
        m_ = copy.deepcopy(m.reshape(-1, window_size))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, num_feat, d)
        
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

        # perform cross-attention
        x = self.cross_attention(x, m)

        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)

        # linear projection
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        x = m*x_inp + (1-m)*x
        
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
        
        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
        # linear projection
        x = self.projection(x)
        
        x = m*x_inp + (1-m)*x

        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        
        return x
 
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
        
        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed(x)

        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
        # linear projection
        x = self.projection(x)
        
        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        
        return x