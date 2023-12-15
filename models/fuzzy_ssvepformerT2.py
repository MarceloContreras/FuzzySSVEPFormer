import torch.nn as nn
import torch.nn.functional as F
import torch

from .neurofuzzy import *
from .ssvepformer import Chnl_combination, CNN_module, MLP_head, Encoder

"""
Fuzzy Type 1 blocks
"""

class LowerGaussMembFunc(torch.nn.Module):
    '''
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            beta, the inverse of standard deviation.
    '''
    def __init__(self, mu, sigma, alpha,h):
        super().__init__()
        self.mu    = mu
        self.sigma = sigma
        self.register_parameter('alpha', alpha)
        self.register_parameter('h',h)

    def forward(self, x):
        val = F.sigmoid(self.h)*torch.exp(-torch.pow(x - self.mu, 2) / (2 * (0.5*F.sigmoid(self.alpha)*self.sigma+0.5)**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {} {}'.format(self.mu, self.sigma, self.alpha)


class fuzzyT2_chn_MLP_module(nn.Module):
    def __init__(self, dropout, channels, length):
      super().__init__()
      self.norm       = nn.LayerNorm([2*channels,length])
      self.linear     = nn.Linear(length,length)
      self.dropout    = nn.Dropout(dropout)

      # 1. Membership
      self.rules  = 2*channels
      self.dim    = length
      self.mu     = nn.Parameter(torch.zeros(self.rules,self.dim))
      self.sigma  = nn.Parameter(torch.ones(self.rules,self.dim))
      self.sigma_u= nn.Parameter(0.75*torch.ones(self.rules,self.dim))
      self.h      = nn.Parameter(0.75*torch.ones(self.rules,self.dim))
      self.GaussMF_u = GaussMembFunc(self.mu, self.sigma)
      self.GaussMF_l = LowerGaussMembFunc(self.mu, self.sigma, self.sigma_u, self.h)

      # 2. Inference
      self.Lambda_u = LambdaLayer(lambda x: tnorm_log_prod(x))
      self.Lambda_l = LambdaLayer(lambda x: tnorm_log_prod(x))

      # 3. Consequent
      self.dense_consequent = nn.Linear(length,self.dim)

      # 4. Fusion Lower and upper Fuzzy T1
      self.beta = nn.Parameter(0.5*torch.ones(1)) # Scalar

    def MLP_block(self,x):
      x = self.norm(x)
      memberships_u = self.GaussMF_u(x)
      memberships_l = self.GaussMF_l(x)
      consequent_u  = torch.mul(self.dense_consequent(x),memberships_u)
      consequent_l  = torch.mul(self.dense_consequent(x),memberships_l)
      output_fuzzy  = (1-F.sigmoid(self.beta))*(consequent_u)+F.sigmoid(self.beta)*consequent_l
      x = self.linear(x) + output_fuzzy
      x = self.dropout(F.gelu(x))
      return x

    def forward(self,x): return x + self.MLP_block(x)


class fuzzyT2MLP_head(nn.Module):
    def __init__(self, channel, classes, length, dropout):
        super(fuzzyT2MLP_head, self).__init__()

        # Layers
        self.flatten = nn.Flatten()
        self.drop1   = nn.Dropout(dropout)
        self.dense1  = nn.Linear(2*channel*length,6*classes)
        self.norm    = nn.LayerNorm([6*classes])
        self.drop2   = nn.Dropout(dropout)
        self.dense2  = nn.Linear(6*classes,classes)

        #Membership

        # 1. Antecedent
        self.rules  = classes
        self.dim    = 6*classes
        self.mu     = nn.Parameter(torch.zeros(self.rules,self.dim))
        self.sigma  = nn.Parameter(torch.ones(self.rules,self.dim))
        self.sigma_u= nn.Parameter(0.75*torch.ones(self.rules,self.dim))
        self.h      = nn.Parameter(0.75*torch.ones(self.rules,self.dim))
        self.GaussMF_u = GaussMembFunc(self.mu, self.sigma)
        self.GaussMF_l = LowerGaussMembFunc(self.mu, self.sigma, self.sigma_u, self.h)

        # 2. Inference
        self.Lambda_u = LambdaLayer(lambda x: tnorm_log_prod(x))
        self.Lambda_l = LambdaLayer(lambda x: tnorm_log_prod(x))

        # 3. Consequent
        self.dense_consequent = nn.Linear(6*classes,classes)

        # 4. Fusion Lower and upper Fuzzy T1
        self.beta = nn.Parameter(0.5*torch.ones(1)) # Scalar


    def forward(self, x):
        x = self.drop1(self.flatten(x))
        x = self.norm(self.dense1(x))
        x = self.drop2(F.gelu(x))
        memberships_u  = self.GaussMF_u(x.unsqueeze(dim=1))
        rules_lambda_u = self.Lambda_u(memberships_u)
        rules_lambda_u = rules_lambda_u.squeeze(0)
        memberships_l  = self.GaussMF_l(x.unsqueeze(dim=1))
        rules_lambda_l = self.Lambda_l(memberships_l)
        rules_lambda_l = rules_lambda_l.squeeze(0)
        consequent_u = torch.mul(self.dense_consequent(x),rules_lambda_u)
        consequent_l = torch.mul(self.dense_consequent(x),rules_lambda_l)
        output_fuzzy   = (1-F.sigmoid(self.beta))*(consequent_u)+F.sigmoid(self.beta)*consequent_l
        x = self.dense2(x) + output_fuzzy
        return x
    

class fuzzyT2Encoder(nn.Module):
    def __init__(self, dropout, channels, length):
      super().__init__()
      self.metaencoder = nn.Sequential(CNN_module(dropout, channels, length),
                                       fuzzyT2_chn_MLP_module(dropout, channels, length),
                                       CNN_module(dropout, channels, length),
                                       fuzzyT2_chn_MLP_module(dropout, channels, length))

    def forward(self,x):
      return self.metaencoder(x)


# Variant A
class fuzzyT2SSVEPformerA(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = Encoder(dropout, channels, length)
      self.MLP = fuzzyT2MLP_head(channels, classes, length, dropout)

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
class fuzzyT2SSVEPformerB(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = fuzzyT2Encoder(dropout, channels, length)
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
class fuzzyT2SSVEPformerC(nn.Module):
    def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
      super().__init__()
      self.chn_combination = Chnl_combination(dropout, channels, length)
      self.SSVEPencoder = fuzzyT2Encoder(dropout, channels, length)
      self.MLP = fuzzyT2MLP_head(channels, classes, length, dropout)

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
