import torch.nn as nn
import torch.nn.functional as F
import torch

"""
SSVEPformer
"""


class Chnl_combination(nn.Module):
    def __init__(self, dropout, channels, length):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2 * channels, 1, padding=1 // 2, groups=1)
        self.norm = nn.LayerNorm(length)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(self.conv(x))
        x = self.drop(F.gelu(x))
        return x


class CNN_module(nn.Module):
    def __init__(self, dropout, channels, length):
        super().__init__()
        self.norm1 = nn.LayerNorm(length)
        self.conv = nn.Conv1d(2 * channels, 2 * channels, 31, padding=31 // 2, groups=1)
        self.norm2 = nn.LayerNorm(length)
        self.dropout = nn.Dropout(dropout)

    def convblock(self, x):
        x = self.conv(self.norm1(x))
        x = F.gelu(self.norm2(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        return x + self.convblock(x)


class chn_MLP_module(nn.Module):
    def __init__(self, dropout, length):
        super().__init__()
        self.norm = nn.LayerNorm(length)
        self.linear = nn.Linear(length, length)
        self.dropout = nn.Dropout(dropout)

    def MLP_block(self, x):
        x = self.linear(self.norm(x))
        x = self.dropout(F.gelu(x))
        return x

    def forward(self, x):
        return x + self.MLP_block(x)


class Encoder(nn.Module):
    def __init__(self, dropout, channels, length):
        super().__init__()
        self.metaencoder = nn.Sequential(
            CNN_module(dropout, channels, length),
            chn_MLP_module(dropout, length),
            CNN_module(dropout, channels, length),
            chn_MLP_module(dropout, length),
        )

    def forward(self, x):
        return self.metaencoder(x)


class MLP_head(nn.Module):
    def __init__(self, channel, classes, length, dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(2 * channel * length, 6 * classes)
        self.norm = nn.LayerNorm(6 * classes)
        self.drop2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(6 * classes, classes)

    def forward(self, x):
        x = self.drop1(self.flatten(x))
        x = self.norm(self.dense1(x))
        x = self.dense2(self.drop2(F.gelu(x)))
        return x


class SSVEPformer(nn.Module):
    def __init__(self, params, signal_size, dropout=0.5):
        super().__init__()

        self.nfft = round(params["Fs"] / params["Resolution"])
        self.fft_start = int(round(params["Filt.low_cut"] / params["Resolution"]))
        self.fft_end = int(round(params["Filt.high_cut"] / params["Resolution"])) + 1
        length = int(2 * (self.fft_end - self.fft_start - 1))
        # length = 2*round(signal_size*params['Fs'])
        # self.nfff = length

        self.chn_combination = Chnl_combination(dropout, params["Channels"], length)
        self.SSVEPencoder = Encoder(dropout, params["Channels"], length)
        self.MLP = MLP_head(params["Channels"], params["Classes"], length, dropout)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # fft_im = torch.fft.fft(x)/self.nfff
        # real = fft_im.real
        # imag = fft_im.imag
        # x = torch.cat((real,imag), -1)
        fft_im = torch.fft.fft(x, n=self.nfft, dim=-1) / (self.nfft / 2)
        real = fft_im.real
        imag = fft_im.imag
        x = torch.cat(
            (
                real[..., self.fft_start : self.fft_end - 1],
                imag[..., self.fft_start : self.fft_end - 1],
            ),
            -1,
        )
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
