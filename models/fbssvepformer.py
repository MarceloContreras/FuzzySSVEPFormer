import torch.nn as nn
import torch

"""
Assuming three filters in FB
"""


class FBSSVEPformer(nn.Module):
    def __init__(self, net1, net2, net3, n_classes):
        super().__init__()
        self.subnet1 = net1
        self.subnet2 = net2
        self.subnet3 = net3
        self.fusion_layer = nn.Linear(3 * n_classes, n_classes)

    def forward(self, x):
        x1 = self.subnet1(x[:, 0, ...])
        x2 = self.subnet2(x[:, 1, ...])
        x3 = self.subnet3(x[:, 2, ...])
        x = self.fusion_layer(torch.cat((x1, x2, x3), -1))
        return x
