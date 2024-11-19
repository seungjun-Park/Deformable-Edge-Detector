import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, Union, Dict, Iterable
from torch.fft import irfftn, rfftn

from utils import to_ntuple, to_2tuple, to_1tuple, to_3tuple


class ComplexConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple] = 1,
                 padding: Union[int, List, Tuple] = 0,
                 dilation: Union[int, List, Tuple] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        self.dtype = dtype
        self.stride = to_1tuple(stride)
        self.dilation = to_1tuple(dilation)
        self.padding = to_1tuple(padding)
        self.padding_mode = padding_mode.lower()
        self.groups = groups
        self.bias = bias
        if dtype == torch.complex32:
            self.weight_dtype = torch.float16
        elif dtype == torch.complex64:
            self.weight_dtype = torch.float32
        else:
            self.weight_dtype = torch.float64

        self.weight = nn.Parameter(torch.ones(out_channels * 2, (in_channels * 2) // groups, *to_1tuple(kernel_size),
                                              device=device, dtype=self.weight_dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * 2, device=device, dtype=self.weight_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        assert len(x.shape) == 3
        x_r, x_i = x.real, x.imag
        x = torch.cat([x_r, x_i], dim=1)
        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                     groups=self.groups)
        x_r, x_i = torch.chunk(x, 2, dim=1)
        x = torch.complex(x_r, x_i)

        return x


class ComplexConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple] = 1,
                 padding: Union[int, List, Tuple] = 0,
                 dilation: Union[int, List, Tuple] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        self.dtype = dtype
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.padding = to_2tuple(padding)
        self.padding_mode = padding_mode.lower()
        self.groups = groups
        self.bias = bias
        if dtype == torch.complex32:
            self.weight_dtype = torch.float16
        elif dtype == torch.complex64:
            self.weight_dtype = torch.float32
        else:
            self.weight_dtype = torch.float64

        self.weight = nn.Parameter(torch.ones(out_channels * 2, (in_channels * 2) // groups, *to_2tuple(kernel_size),
                                              device=device, dtype=self.weight_dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * 2, device=device, dtype=self.weight_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        assert len(x.shape) == 4
        x_r, x_i = x.real, x.imag
        x = torch.cat([x_r, x_i], dim=1)
        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        x_r, x_i = torch.chunk(x, 2, dim=1)
        x = torch.complex(x_r, x_i)

        return x


class ComplexConv3d(nn.Module):
    def __init__(self):
        super().__init__()

