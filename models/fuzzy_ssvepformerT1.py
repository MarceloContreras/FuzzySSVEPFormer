import torch.nn as nn
import torch.nn.functional as F
import torch

from .neurofuzzy import *
from .ssvepformer import Chnl_combination, CNN_module, MLP_head, Encoder

"""
Fuzzy Type 1 blocks
"""

class fuzzy_chn_MLP_module(nn.Module):
    def __init__(self, dropout, channels, length):
      super().__init__()
      self.norm       = nn.LayerNorm([2*channels,length])
      self.linear     = nn.Linear(length,length)
      self.dropout    = nn.Dropout(dropout)

      #Membership
      self.rules  = 2*channels
      self.dim    = length
      self.dense_consequent = nn.Linear(length,self.dim)
      self.mu     = nn.Parameter(torch.zeros(self.rules,self.dim))
      self.beta   = nn.Parameter(torch.ones(self.rules,self.dim))
      self.GaussMF = GaussMembFunc(self.mu, self.beta)
      self.Lambda = LambdaLayer(lambda x: tnorm_log_prod(x))

    def MLP_block(self,x):
      x = self.norm(x) 
      rules_lambda = self.GaussMF(x)
      x = self.linear(x) + torch.mul(self.dense_consequent(x),rules_lambda)
      x = self.dropout(F.gelu(x))
      return x

    def forward(self,x): return x + self.MLP_block(x)

class fuzzyMLP_head(nn.Module):
    def __init__(self, channel, classes, length, dropout):

      super(fuzzyMLP_head, self).__init__()

      # Layers
      self.flatten = nn.Flatten()
      self.drop1   = nn.Dropout(dropout)
      self.dense1  = nn.Linear(2*channel*length,6*classes)
      self.norm    = nn.LayerNorm([6*classes])
      self.drop2   = nn.Dropout(dropout)
      self.dense2  = nn.Linear(6*classes,classes)
      self.dense_consequent = nn.Linear(6*classes,classes)

      #Membership
      self.rules = classes
      self.dim   = 6*classes
      self.mu    = nn.Parameter(torch.zeros(self.rules,self.dim))
      self.sigma = nn.Parameter(torch.ones(self.rules,self.dim))
      self.GaussMF = GaussMembFunc(self.mu, self.sigma)
      self.Lambda  = LambdaLayer(lambda x: tnorm_log_prod(x))

    def forward(self, x):
      x = self.drop1(self.flatten(x))
      x = self.norm(self.dense1(x))
      x = self.drop2(F.gelu(x))

      memberships = self.GaussMF(x.unsqueeze(dim=1))
      rules_lambda = self.Lambda(memberships)
      rules_lambda = rules_lambda.squeeze(0)
      x = self.dense2(x) +  torch.mul(self.dense_consequent(x),rules_lambda)
      return x
    

class fuzzyEncoder(nn.Module):
    def __init__(self, dropout, channels, length):
      super().__init__()
      self.metaencoder = nn.Sequential(CNN_module(dropout, channels, length),
                                       fuzzy_chn_MLP_module(dropout, channels, length),
                                       CNN_module(dropout, channels, length),
                                       fuzzy_chn_MLP_module(dropout, channels, length))

    def forward(self,x):
      return self.metaencoder(x)


# Variant A
class fuzzySSVEPformerA(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = Encoder(dropout, channels, length)
      self.MLP = fuzzyMLP_head(channels, classes, length, dropout)

    def forward(self,x):
      fft_im = torch.fft.fft(x)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real,imag), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x

# Variant B
class fuzzySSVEPformerB(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = fuzzyEncoder(dropout, channels, length)
      self.MLP = MLP_head(channels, classes, length, dropout)

    def forward(self,x):
      fft_im = torch.fft.fft(x)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real,imag), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x
    
# Variant C
class fuzzySSVEPformerC(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = fuzzyEncoder(dropout, channels, length)
      self.MLP = fuzzyMLP_head(channels, classes, length, dropout)

    def forward(self,x):
      fft_im = torch.fft.fft(x)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real,imag), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x


def init_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.normal_(m.weight,0,0.01)