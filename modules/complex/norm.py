import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple


class ComplexBatchNorm(nn.Module):
    def __init__(self,
                 channels: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.channels = channels
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels, dtype=dtype))
            self.shift = nn.Parameter(torch.zeros(channels), dtype=dtype)

        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels, dtype=dtype))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype
        b, c, *_ = x.shape

        assert self.channels == c

        x = x.reshape(b, c, -1)

        if self.training or not self.track_running_stats:
            mean = torch.mean(x, dim=[0, 2])
            var = torch.var(x, dim=[0, 2])

            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var

        else:
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x - mean.reshape(1, -1, 1)) / torch.sqrt(var + self.eps).reshape(1, -1, 1)

        if self.affine:
            x_norm = self.scale.reshape(1, -1, 1) * x_norm + self.shift.reshape(1, -1, 1)

        return x_norm.reshape(b, c, *_)


class ComplexLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape: Union[int, List[int], Tuple[int], torch.Size],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 dtype=torch.complex64,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, List):
            normalized_shape = torch.Size([normalized_shape])
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype

        assert self.normalized_shape == x.shape[-len(self.normalized_shape): ]

        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm


class ComplexInstanceNorm(nn.Module):
    def __init__(self,
                 channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 dtype=torch.complex64,
                 *args,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.channels = channels
        self.dtype = dtype
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels, dtype=dtype))
            self.shift = nn.Parameter(torch.zeros(channels, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype

        b, c, *_ = x.shape

        assert c == self.channels

        x = x.reshape(b, c, -1)

        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, dim=[-1], keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.reshape(b, c, -1)

        if self.affine:
            x_norm = self.scale.reshape(1, -1, 1) * x_norm + self.shift.reshape(1, -1, 1)

        return x_norm.reshape(b, c, *_)


class ComplexGroupNorm(nn.Module):
    def __init__(self,
                 channels: int,
                 groups: int = 32,
                 eps: float = 1e-5,
                 affine: bool = True,
                 dtype=torch.complex64,
                 *args,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        assert channels % groups == 0

        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.affine = affine
        self.dtype = dtype

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels, dtype=dtype))
            self.shift = nn.Parameter(torch.zeros(channels, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype

        b, c, *_ = x.shape

        assert c == self.channels

        x = x.reshape(b, self.groups, -1)

        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, dim=[-1], keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm.reshape(b, c, -1)
            x_norm = self.scale.reshape(1, -1, 1) * x_norm + self.shift.reshape(1, -1, 1)

        return x_norm.view(b, c, *_)
