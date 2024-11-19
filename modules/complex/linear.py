import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple


class ComplexLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Union[torch.device, str] = None,
                 dtype: torch.dtype = torch.complex64,
                 *args,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        if dtype == torch.complex32:
            self.weight_dtype = torch.float16
        elif dtype == torch.complex64:
            self.weight_dtype = torch.float32
        else:
            self.weight_dtype = torch.float64

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones((out_features * 2, in_features * 2), device=device, dtype=self.weight_dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features * 2, device=device, dtype=self.weight_dtype))

        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        x_r, x_i = x.real, x.imag
        x = torch.cat([x_r, x_i], dim=-1)
        x = F.linear(x, self.weight, self.bias)
        x_r, x_i = torch.chunk(x, 2, dim=-1)
        x = torch.complex(x_r, x_i)

        return x
