from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'wave_T': _cfg(crop_pct=0.9),
    'wave_S': _cfg(crop_pct=0.9),
    'wave_M': _cfg(crop_pct=0.9),
    'wave_B': _cfg(crop_pct=0.875),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode=mode
        
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    


    def forward(self, x):
     
        B, C, H, W = x.shape
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)

        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)

#         x_1=self.fc_h(x)
#         x_2=self.fc_w(x)
#         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
#         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
        
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x
        
class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x


class PatchEmbedOverlapping(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d, groups=1,use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)      
        self.norm = norm_layer(embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size,norm_layer=nn.BatchNorm2d,use_norm=True):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_layer(out_embed_dim) if use_norm==True else nn.Identity()
    def forward(self, x):
        x = self.proj(x) 
        x = self.norm(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.,norm_layer=nn.BatchNorm2d,mode='fc', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer,mode=mode))
    blocks = nn.Sequential(*blocks)
    return blocks

class WaveNet(nn.Module):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, mlp_ratios=None, 
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.BatchNorm2d, fork_feat=False,mode='fc',ds_use_norm=True,args=None): 

        super().__init__()
        
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0],norm_layer=norm_layer,use_norm=ds_use_norm)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer,mode=mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size,norm_layer=norm_layer,use_norm=ds_use_norm))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1]) 
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None): 
        """ mmseg or mmdet `init_weight` """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        cls_out = self.head(F.adaptive_avg_pool2d(x,output_size=1).flatten(1))#x.mean(1)
        return cls_out

def MyNorm(dim):
    return nn.GroupNorm(1, dim)    
    
@register_model
def WaveMLP_T_dw(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,mode='depthwise', **kwargs)
    model.default_cfg = default_cfgs['wave_T']
    return model    
    
@register_model
def WaveMLP_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['wave_T']
    return model

@register_model
def WaveMLP_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm, **kwargs)
    model.default_cfg = default_cfgs['wave_S']
    return model

@register_model
def WaveMLP_M(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['wave_M']
    return model

@register_model
def WaveMLP_B(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 18, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = WaveNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['wave_B']
    return model
    
pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self,input_dim=4,inner_dim=128,output_dim=4,dropout=0.):
        super().__init__()
        self.fn = nn.Sequential(
        nn.Conv2d(input_dim, inner_dim,3,1,1,groups=input_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(inner_dim, output_dim,1,1,0,groups=output_dim),
        nn.Dropout(dropout)
    )
        self.norm = nn.LayerNorm(208)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
    
    
class Ourmodel(nn.Module):
    def __init__(self, ):
        super(Ourmodel, self).__init__()
        self.am1 = nn.Sequential(
        PreNormResidual(4,512,4),
        PreNormResidual(4,512,4),
        PreNormResidual(4,512,4)
    )
        self.am2 = nn.Sequential(
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4))
        
        self.ph1 = nn.Sequential(
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4))
        self.ph2 = nn.Sequential(
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4),
                PreNormResidual(4,256,4)
                )
        
        self.comb = nn.Sequential(WaveBlock(256),WaveBlock(256),WaveBlock(256),WaveBlock(256),WaveBlock(256),WaveBlock(256),WaveBlock(256))#batch 8 , 4 , 208 want to get a batch 8,32, 208
        self.adjust = nn.Conv2d(256,64,1,1,0)
        self.Linear = nn.Linear(208,64)
        
        self.exp = nn.Conv2d(8,256,1,1,0)
    def forward(self, x):
        x = x.transpose(-1,-2)
        #print(x.shape,"x")
        x1 = x[:,:1,:,:,:].squeeze(1)
        x2 = x[:,1:,:,:,:].squeeze(1)
        #print(x1.shape,"x1")
        x1 = self.am1(x1)
        x1 = self.am1(x1.transpose(1,2))
        
        x2 = self.ph1(x2)
        x2 = self.ph2(x2.transpose(1,2))
        
        x_comb = torch.cat((x1,x2), axis=1) # x_comb ---> dimension ====8
        x_comb = self.exp(x_comb)
        #print(x_comb.shape,"x_comb") #batch 256 4 48
        x_comb = self.adjust(self.comb(x_comb))#.transpose(1,2) #  batchsize 64 4 48
        a,b,c,d = x_comb.shape[0],x_comb.shape[1],x_comb.shape[2],x_comb.shape[3]#batchsize 64 4 48
        #print(x_comb.shape,a)
        x_comb = x_comb.reshape(a,-1,208)
        x_combout = self.Linear(x_comb).view(a,2,4,-1,64)
        #print(x_combout.shape,"x_combout")
        return x_combout # must be 2,4,32,64
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DACEN.py
@Time    :   2023/02/28 14:19:53
@Author  :   Binggui ZHOU
@Version :   1.0
@Contact :   binggui.zhou[AT]connect.um.edu.mo
@License :   (C)Copyright 2018-2023, UM, SKL-IOTSC, MACAU, CHINA
@Desc    :   None
'''
import math
import torch
from torch import nn
from torch.utils.data import Dataset
from einops import rearrange

#=======================================================================================================================
#=======================================================================================================================
# DataLoader Defining

class DatasetFolder(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel
    def __getitem__(self, index):
        return self.input[index], self.label[index]
    def __len__(self):
        return self.input.shape[0]

class DatasetFolder_weights(Dataset):
    def __init__(self, matInput, matLabel, weights):
        self.input, self.label, self.weights = matInput, matLabel, weights
    def __getitem__(self, index):
        return self.input[index], self.label[index], self.weights[index], 
    def __len__(self):
        return self.label.shape[0]

class DatasetFolder_eval(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

#=======================================================================================================================
#=======================================================================================================================
# Module and Model Defining

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, 'b (rx tx) dmodel -> b dmodel rx tx', rx=4)
        _x = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x) * _x
        x = rearrange(x, 'b dmodel rx tx -> b (rx tx) dmodel')
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SpatialAttentionModule(nn.Module):

    def __init__(self, d_model, ffn_hidden, drop_prob):
        super(SpatialAttentionModule, self).__init__()
        self.attention = SpatialAttention(kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):

        _x = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x

class DualAttentionChannelEstimationNetwork_Our(nn.Module):
    def __init__(self, input_length):
        super(DualAttentionChannelEstimationNetwork_Our, self).__init__()
        pilot_num = int(input_length / 8)
        
        d_model = 512
        d_hid = 512
        dropout = 0.0
        nlayers = 2
        self.fc1 = nn.Linear(2*pilot_num, d_model)
        self.sa_layers = nn.ModuleList([SpatialAttentionModule(d_model=d_model,
                                                  ffn_hidden=d_hid,
                                                  drop_prob=dropout)
                                     for _ in range(nlayers)])
        self.fc2 = nn.Linear(d_model, 64*2)
        

        d_model = 512
        nhead = 2
        d_hid = 512
        dropout = 0.0
        nlayers = 2
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.ta_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
                                     for _ in range(nlayers)])
        self.fc3 = nn.Linear(4*32, d_model)

        self.fc4 = nn.Linear(d_model, 4*32)
        self.expa = nn.Conv2d(8,56,1,1,0)
        self.expa2 = nn.Conv2d(56,8,1,1,0)
        self.wave = nn.Sequential(WaveBlock(56),WaveBlock(56),WaveBlock(56))

    def forward(self, x):
        out = rearrange(x, 'b c rx (rb subc) sym -> b (rx subc sym) (c rb)', subc=8)
        out_temp_all = self.fc1(out)
        out_temp = out_temp_all
        out_temp2 = out_temp_all
        #print(out.shape,"1-----") #batch,128,516
        
        for layer in self.sa_layers:
            out_temp = layer(out_temp)
        #print(out.shape,"2-----")#batch,128,516
        out_temp = self.fc2(out_temp)
        out_temp = rearrange(out_temp, 'b (rx tx) (c d) -> (c d) b (rx tx)', c=2, rx=4)
        out_temp = self.fc3(out_temp)
        out_temp = self.pos_encoder(out_temp)
        #print(out_temp.shape,"3-----")#128, batch,516
        
        for layer in self.ta_layers:
            out_temp2 = layer(out_temp2)
        #print(out_temp2.shape,"4-----")#128, batch,516
        
        
        
        out = out_temp2.transpose(0,1)+out_temp
        out = self.fc4(out)
        #print(out.shape,"5-----")#128, batch,128
        out = rearrange(out, '(c d) b (rx tx) -> b (c rx) tx d', c=2, rx=4)
        #print(out.shape,"out111111")
        out = self.expa(out)
        out = self.wave(out)
        out = self.expa2(out)
        out = rearrange(out, 'b (c rx) tx d -> b c rx tx d', c=2, rx=4)
        #print(out.shape,"6-----")# batch,2,4,32,64
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        


#model  = Ourmodel()
#print(model(torch.ones(16,2,4,4,208)).shape)