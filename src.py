import pandas as pd
import numpy as np
import zipfile
import torch
from functools import lru_cache
import os
from copy import deepcopy
from synergy.combination import MuSyC, BRAID, Zimmer
from utils import *
from torchmetrics import AUROC
from functools import lru_cache
from torch import nn

class LatentHillFusionModule(nn.Module):
    def __init__(self, init_dim, hidden_dim, dropout=0.0, use_norm_slope = True, use_norm_bias = True):
        super().__init__()
        self.use_norm_bias = use_norm_bias
        self.use_norm_slope = use_norm_slope
        if use_norm_slope:
            norm11 = nn.LayerNorm(hidden_dim)
            norm12 = nn.LayerNorm(init_dim)
        else:
            norm11 = nn.Identity()
            norm12 = nn.Identity()
        if use_norm_bias:
            norm21 = nn.LayerNorm(hidden_dim)
            norm22 = nn.LayerNorm(init_dim)
        else:
            norm21 = nn.Identity()
            norm22 = nn.Identity()
        self.norm11 = norm11
        self.norm21 = norm21
        self.mlp_bias = nn.Sequential(nn.LazyLinear(hidden_dim),
                                      norm11,
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm12)
        self.mlp_slope = nn.Sequential(nn.LazyLinear(hidden_dim),
                                      norm21,
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm22)
    def forward(self, c, drugs):
        bias = self.mlp_bias(c.squeeze())
        for i, (d, conc) in enumerate(drugs):
            d_slope = self.mlp_slope(torch.cat([d.squeeze(), c.squeeze()], -1))
            if i == 0:
                slope = conc.unsqueeze(2)*d_slope.unsqueeze(1)
            else:
                slope += (conc.unsqueeze(2)*d_slope.unsqueeze(1))
        h = torch.sigmoid(bias.unsqueeze(-2) + slope)
        return h
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
            
class LatentHillFusionModuleDrugEmbedding(nn.Module):
    def __init__(self, init_dim, hidden_dim, dropout=0.0, use_norm_slope = True, use_norm_bias = True):
        super().__init__()
        self.use_norm_bias = use_norm_bias
        self.use_norm_slope = use_norm_slope
        if use_norm_slope:
            norm11 = nn.LayerNorm(hidden_dim)
            norm12 = nn.LayerNorm(init_dim)
        else:
            norm11 = nn.Identity()
            norm12 = nn.Identity()
        if use_norm_bias:
            norm21 = nn.LayerNorm(hidden_dim)
            norm22 = nn.LayerNorm(init_dim)
        else:
            norm21 = nn.Identity()
            norm22 = nn.Identity()
        self.norm11 = norm11
        self.norm21 = norm21
        self.mlp_bias = nn.Sequential(nn.LazyLinear(hidden_dim),
                                      norm11,
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm12)
        self.mlp_slope = nn.Sequential(nn.LazyLinear(hidden_dim),
                                      norm21,
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm22)
    def forward(self, c, drugs):
        bias = self.mlp_bias(c)
        for i, (d, conc) in enumerate(drugs):
            d_slope = self.mlp_slope(torch.cat([d, c], -1))
        return d_slope
    
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
            

class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers = 6, layernorm = True):
        super().__init__()
        self.mlps = nn.ModuleList()
        for l in range(n_layers):
            self.mlps.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, embed_dim)))
        self.lin = nn.Linear(embed_dim, 1)
    def forward(self, x):
        for l in self.mlps:
            x = l(x) + x
        return self.lin(x)
        
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()

            
class Model(nn.Module):
    def __init__(self, embed_dim=256,
                 hidden_dim_fusion=1024,
                 hidden_dim_mlp=1024,
                 dropout_fusion=0.1,
                 dropout_res=0.1,
                 use_norm_bias = False,
                 use_norm_slope = False,
                 num_res=2):
        super().__init__()
        self.fusion = LatentHillFusionModule(embed_dim,
                                             hidden_dim_fusion,
                                             dropout=dropout_fusion, 
                                             use_norm_bias=use_norm_bias,
                                             use_norm_slope = use_norm_slope)
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_fusion))
        self.embed_c = nn.Embedding(num_embeddings=270, embedding_dim=embed_dim)
        self.mlp = ResNet(embed_dim, hidden_dim_mlp, dropout_res, num_res)
    def forward(self, c, drugs):
        ds = []
        for drug,conc in drugs:
            ds += [(self.embed_d(drug), conc.flatten(-2))]
        c = self.embed_c(c).squeeze(1)
        B, n1, n2 = drugs[0][1].shape
        return self.mlp(self.fusion(c, ds)).reshape(B, n1, n2)
        
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()

class ModelEmbedding(nn.Module):
    def __init__(self, embed_dim=256,
                 hidden_dim_fusion=1024,
                 hidden_dim_mlp=1024,
                 dropout_fusion=0.1,
                 dropout_res=0.1,
                 use_norm_bias = False,
                 use_norm_slope = False,
                 num_res=2):
        super().__init__()
        self.fusion = LatentHillFusionModuleDrugEmbedding(embed_dim,
                                             hidden_dim_fusion,
                                             dropout=dropout_fusion, 
                                             use_norm_bias=use_norm_bias,
                                             use_norm_slope = use_norm_slope)
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_fusion))
        self.embed_c = nn.Embedding(num_embeddings=70, embedding_dim=embed_dim)
        self.mlp = ResNet(embed_dim, hidden_dim_mlp, dropout_res, num_res)
    def forward(self, c, drugs):
        ds = []
        fps = []
        embeddings = []
        for drug,conc in drugs:
            fps += [drug]
            drug_embed = self.embed_d(drug)
            embeddings += [drug_embed]
            ds += [(drug_embed, conc.flatten(-2))]
        c = self.embed_c(c).squeeze(1)
        B, n1, n2 = drugs[0][1].shape
        interactions = self.fusion(c, ds)
        return torch.stack(fps), torch.stack(embeddings), interactions
        
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
            
class ModelCombo(nn.Module):
    def __init__(self, embed_dim=256,
                 hidden_dim_fusion=1024,
                 hidden_dim_mlp=1024,
                 dropout_fusion=0.1,
                 dropout_res=0.1,
                 use_norm_bias = False,
                 use_norm_slope = False,
                 num_res=2):
        super().__init__()
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_fusion))
        self.embed_c = nn.Embedding(num_embeddings=70, embedding_dim=embed_dim)
        self.mlp = ResNet(embed_dim, hidden_dim_mlp, dropout_res, num_res)
    def forward(self, d1, d2, c):
        c = self.embed_c(c).squeeze(1)
        d1 = self.embed_d(d1).squeeze(1)
        d2 = self.embed_d(d2).squeeze(1)
        return self.mlp(d1 + d2 + c)
        
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
