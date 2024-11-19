import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        x_real = F.relu(x.real)
        x_imag = F.relu(x.imag)

        return torch.complex(x_real, x_imag)


class CLeakReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        x_real = F.leaky_relu(x.real)
        x_imag = F.leaky_relu(x.imag)

        return torch.complex(x_real, x_imag)


class CSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        x_real = F.silu(x.real)
        x_imag = F.silu(x.imag)

        return torch.complex(x_real, x_imag)


class CGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        x_real = F.gelu(x.real)
        x_imag = F.gelu(x.imag)

        return torch.complex(x_real, x_imag)


class ComplexSoftmax(nn.Module):
    def __init__(self,
                 dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert torch.is_complex(x)
        if self.dim is None:
            ndim = x.dim()
            if ndim == 0 or ndim == 1 or ndim == 3:
                self.dim = 0
            else:
                self.dim = 1

        exp_x = torch.exp(x.real) * (torch.cos(x.imag) + 1j * torch.sin(x.imag))
        softmax_prob = exp_x / (torch.sum(exp_x, dim=self.dim, keepdim=True))

        return softmax_prob
