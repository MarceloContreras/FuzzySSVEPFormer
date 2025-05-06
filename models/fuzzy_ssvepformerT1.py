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
      self.norm       = nn.LayerNorm(length)
      self.linear     = nn.Linear(length,length)
      self.dropout    = nn.Dropout(dropout)

      #Membership
      self.rules = 2*channels
      self.dim   = length
      self.dense_consequent = nn.Linear(length,self.dim)
      self.mu   = nn.Parameter(torch.zeros(self.rules,self.dim))
      self.beta = nn.Parameter(torch.ones(self.rules,self.dim))
      self.GaussMF = GaussMembFunc(self.mu, self.beta)
      self.Lambda  = LambdaLayer(lambda x: tnorm_log_prod(x))

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
      self.norm    = nn.LayerNorm(6*classes)
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
    def __init__(self, params, signal_size, dropout = 0.5):
      super().__init__()
      self.nfft = round(params['Fs']/params['Resolution']) 
      self.fft_start = int(round(params['Filt.low_cut']/params['Resolution']))
      self.fft_end = int(round(params['Filt.high_cut']/params['Resolution'])) + 1
      length = int(2*(self.fft_end-self.fft_start-1))

      self.chn_combination = Chnl_combination(dropout, params['Channels'], length)
      self.SSVEPencoder = Encoder(dropout, params['Channels'], length)
      self.MLP = fuzzyMLP_head(params['Channels'], params['Classes'], length, dropout)

      for m in self.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self,x):
      fft_im = torch.fft.fft(x, n = self.nfft, dim = -1)/(self.nfft/2)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real[...,self.fft_start:self.fft_end-1],imag[...,self.fft_start:self.fft_end-1]), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x


# Variant B
class fuzzySSVEPformerB(nn.Module):
    def __init__(self, params, signal_size, dropout = 0.5):
      super().__init__()
      self.nfft = round(params['Fs']/params['Resolution']) 
      self.fft_start = int(round(params['Filt.low_cut']/params['Resolution']))
      self.fft_end = int(round(params['Filt.high_cut']/params['Resolution'])) + 1
      length = int(2*(self.fft_end-self.fft_start-1))

      self.chn_combination = Chnl_combination(dropout, params['Channels'], length)
      self.SSVEPencoder = fuzzyEncoder(dropout, params['Channels'], length)
      self.MLP = MLP_head(params['Channels'], params['Classes'], length, dropout)

      for m in self.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self,x):
      fft_im = torch.fft.fft(x, n = self.nfft, dim = -1)/(self.nfft/2)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real[...,self.fft_start:self.fft_end-1],imag[...,self.fft_start:self.fft_end-1]), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x
    

# Variant C
class fuzzySSVEPformerC(nn.Module):
    def __init__(self, params, signal_size, dropout = 0.5):
      super().__init__()
      self.nfft = round(params['Fs']/params['Resolution']) 
      self.fft_start = int(round(params['Filt.low_cut']/params['Resolution']))
      self.fft_end = int(round(params['Filt.high_cut']/params['Resolution'])) + 1
      length = int(2*(self.fft_end-self.fft_start-1))
      
      self.chn_combination = Chnl_combination(dropout, params['Channels'], length)
      self.SSVEPencoder = fuzzyEncoder(dropout, params['Channels'], length)
      self.MLP = fuzzyMLP_head(params['Channels'], params['Classes'], length, dropout)

      for m in self.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self,x):
      fft_im = torch.fft.fft(x, n = self.nfft, dim = -1)/(self.nfft/2)
      real = fft_im.real
      imag = fft_im.imag
      x = torch.cat((real[...,self.fft_start:self.fft_end-1],imag[...,self.fft_start:self.fft_end-1]), -1)
      x = self.chn_combination(x)
      x = self.SSVEPencoder(x)
      x = self.MLP(x)
      return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()