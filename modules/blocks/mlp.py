import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_act, conv_nd, to_ntuple, group_norm
from utils.checkpoints import checkpoint


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        embed_dim = in_channels if embed_dim is None else embed_dim
        out_channels = in_channels if out_channels is None else out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint

        self.fc1 = nn.Linear(in_channels, embed_dim)
        self.fc2 = nn.Linear(embed_dim, out_channels)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_channels)

        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x):
        # x.shape == b, l, c
        h = x
        h = self.fc1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout)

        h = self.fc2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout)

        return h


class ConvMLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 num_groups: int = 1,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        embed_dim = in_channels if embed_dim is None else embed_dim
        out_channels = in_channels if out_channels is None else out_channels
        self.use_checkpoint = use_checkpoint

        self.conv1 = conv_nd(dim, in_channels, embed_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_nd(dim, embed_dim, out_channels, kernel_size=1, stride=1, padding=0)

        self.act = get_act(act)

        self.dropout = dropout

        self.norm = group_norm(in_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == b, c, *...

        h = x
        h = self.norm(h)
        h = self.conv1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = F.dropout(h, p=self.dropout)

        return h



