import torch
import torch.nn as nn

class GaussMembFunc(torch.nn.Module):
    '''
        Logarithmic version of Gaussian membership functions, defined by two parameters:
        mu, the mean (center)
    '''
    def __init__(self, mu, sigma):
        super().__init__()
        self.register_parameter('mu', mu)
        self.register_parameter('sigma', sigma)

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)


def tnorm_log_prod(x):
      x = torch.sum(torch.log(x + 1e-8), axis=-1)
      x = x - torch.max(x, axis=1, keepdim=True).values
      return torch.exp(x)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)