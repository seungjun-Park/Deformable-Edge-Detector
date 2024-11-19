import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self,
                 eps=30.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.register_buffer('eps', torch.tensor([eps]))

    def forward(self, x: torch.Tensor):
        return torch.sin(30 * x)


class Cosine(nn.Module):
    def __init__(self,
                 eps=30.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.eps = eps

    def forward(self, x):
        return torch.cos(x)
