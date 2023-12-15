import torch.nn as nn
import torch.nn.functional as F
import torch

"""
SSVEPformer
"""

class Chnl_combination(nn.Module):
    def __init__(self, dropout, channels, length):
      super().__init__()
      self.conv = nn.Conv1d(channels,2*channels,1,padding = 'same')
      self.norm = nn.LayerNorm([channels*2,length])
      self.drop = nn.Dropout(dropout)

    def forward(self,x):
      x = self.norm(self.conv(x))
      x = self.drop(F.gelu(x))
      return x


class CNN_module(nn.Module):
  def __init__(self,dropout, channels, length):
    super().__init__()
    self.norm1   = nn.LayerNorm([channels*2,length])
    self.conv    = nn.Conv1d(2*channels,2*channels,31,padding = 'same')
    self.norm2   = nn.LayerNorm([channels*2,length])
    self.dropout = nn.Dropout(dropout)

  def convblock(self,x):
    x = self.conv(self.norm1(x))
    x = F.gelu(self.norm2(x))
    x = self.dropout(x)
    return x

  def forward(self,x): return x + self.convblock(x)


class chn_MLP_module(nn.Module):
  def __init__(self, dropout, channels, length):
    super().__init__()
    self.norm       = nn.LayerNorm([2*channels,length])
    self.linear     = nn.Linear(length,length)
    self.dropout    = nn.Dropout(dropout)

  def MLP_block(self,x):
    x = self.linear(self.norm(x))
    x = self.dropout(F.gelu(x))
    return x

  def forward(self,x): return x + self.MLP_block(x)


class Encoder(nn.Module):
  def __init__(self, dropout, channels, length):
    super().__init__()
    self.metaencoder = nn.Sequential(CNN_module(dropout, channels, length),
                                     chn_MLP_module(dropout, channels, length),
                                     CNN_module(dropout, channels, length),
                                     chn_MLP_module(dropout, channels, length))

  def forward(self,x):
    return self.metaencoder(x)


class MLP_head(nn.Module):
  def __init__(self, channel, classes, length, dropout):
    super().__init__()
    self.flatten = nn.Flatten()
    self.drop1   = nn.Dropout(dropout)
    self.dense1  = nn.Linear(2*channel*length,6*classes)
    self.norm    = nn.LayerNorm([6*classes])
    self.drop2   = nn.Dropout(dropout)
    self.dense2  = nn.Linear(6*classes,classes)

  def forward(self,x):
    x = self.drop1(self.flatten(x))
    x = self.norm(self.dense1(x))
    x = self.dense2(self.drop2(F.gelu(x)))
    return x


class SSVEPformer(nn.Module):
  def __init__(self, channels = 8, classes = 12, length = 256*2, dropout = 0.5):
    super().__init__()
    self.chn_combination = Chnl_combination(dropout, channels, length)
    self.SSVEPencoder = Encoder(dropout, channels, length)
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


def init_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.normal_(m.weight,0,0.01)

if __name__ == "__main__":
  from ptflops import get_model_complexity_info
  import re

  #Model thats already available
  model = SSVEPformer(channels = 3, classes = 4, length = int(250*2), dropout = 0.5)
  macs, params = get_model_complexity_info(model, (3, 250), as_strings=True,
  print_per_layer_stat=True, verbose=True)
  # Extract the numerical value
  flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
  # Extract the unit
  flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

  print('Computational complexity: {:<8}'.format(macs))
  print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
  print('Number of parameters: {:<8}'.format(params))