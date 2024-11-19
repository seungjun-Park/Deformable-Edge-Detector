import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd, group_norm, get_act
from utils.checkpoints import checkpoint
from modules.blocks.deform_conv import deform_conv_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 num_groups: int = 1,
                 scale_factor: Union[int, float] = 2.0,
                 dim: int = 2,
                 act: str = 'relu',
                 pool_type: str = 'conv',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        scale_factor = int(scale_factor)
        pool_type = pool_type.lower()

        out_channels = in_channels if out_channels is None else out_channels
        if pool_type == 'conv':
            self.pooling = nn.Sequential(
                group_norm(in_channels, num_groups=num_groups),
                conv_nd(
                    dim,
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                )
            )

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(x)

