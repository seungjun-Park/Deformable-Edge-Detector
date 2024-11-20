import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.checkpoints import checkpoint
from timm.models.layers import DropPath
import math

from typing import Union, List, Tuple
from utils import get_act, conv_nd, group_norm, to_2tuple, zero_module, to_ntuple
from modules.blocks.deform_conv import deform_conv_nd


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            group_norm(in_channels, num_groups=num_groups),
            get_act(act),
            conv_nd(dim, in_channels, out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dim, out_channels, out_channels, 3, padding=1)
            )
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 3, padding=1)
            )

        else:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 1),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)

        return self.drop_path(h) + self.shortcut(x)


class DepthWiseSeperableResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            group_norm(in_channels, num_groups=num_groups),
            get_act(act),
            conv_nd(dim, in_channels, in_channels, 3, padding=1, groups=in_channels),
            conv_nd(dim, in_channels, out_channels, 1)
        )

        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dim, out_channels, out_channels, 3, padding=1, groups=out_channels)
            ),
            zero_module(
                conv_nd(dim, out_channels, out_channels, 1)
            )
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, in_channels, 3, padding=1, groups=in_channels),
                conv_nd(dim, in_channels, out_channels, 1)
            )

        else:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 1),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)

        return self.drop_path(h) + self.shortcut(x)


class DeformableResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 dw_kernel_size: int = 7,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        deformable_groups_per_groups = 1
        if deformable_group_channels is not None:
            deformable_groups = math.gcd(in_channels // deformable_group_channels, out_channels // deformable_group_channels)
            deformable_groups_per_groups = (in_channels // deformable_group_channels) // deformable_groups

        self.in_layers = nn.Sequential(
            group_norm(in_channels, num_groups=num_groups),
            get_act(act),
            deform_conv_nd(
                dim,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=deformable_groups,
                deformable_groups_per_groups=deformable_groups_per_groups,
                offset_scale=offset_scale,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size
            )
        )

        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            deform_conv_nd(
                dim,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels // deformable_group_channels if deformable_group_channels is not None else deformable_groups,
                offset_scale=offset_scale,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size
            )
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 3, padding=1),
            )

        else:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 1),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)

        return self.drop_path(h) + self.shortcut(x)
