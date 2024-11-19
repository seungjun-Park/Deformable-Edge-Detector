import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.models.layers import DropPath
from omegaconf import DictConfig
from collections import abc

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img, to_rgb, checkpoint
from modules.blocks.res_block import ResidualBlock, DepthWiseSeperableResidualBlock, DeformableResidualBlock
from modules.blocks.deform_conv import deform_conv_nd
from modules.blocks.down import DownBlock
from modules.blocks.up import UpBlock
from modules.blocks.mlp import ConvMLP
from modules.blocks.norm import GlobalResponseNorm


class SDBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 dw_kernel_size: int = 7,
                 act: str = 'relu',
                 use_conv: bool = True,
                 num_groups: int = 1,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        out_channels = out_channels if out_channels is not None else in_channels

        self.res_block = DeformableResidualBlock(
            in_channels,
            out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            dim=dim,
            deformable_groups=deformable_groups,
            deformable_group_channels=deformable_group_channels,
            offset_scale=offset_scale,
            modulation_type=modulation_type,
            dw_kernel_size=dw_kernel_size,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
            use_conv=use_conv,
        )

        # self.res_block = ResidualBlock(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     dropout=dropout,
        #     drop_path=drop_path,
        #     act=act,
        #     dim=dim,
        #     num_groups=num_groups,
        #     use_checkpoint=use_checkpoint,
        #     use_conv=use_conv
        # )

        self.attn = nn.Sequential(
            group_norm(out_channels, num_groups=num_groups),
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res_block(x)
        h = self.drop_path(self.attn(h)) + h

        return h


class SketchDetectionNetwork(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 embed_dim: int = 16,
                 mlp_ratio: float = 4.0,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 dw_kernel_size: int = 7,
                 num_groups: int = 1,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 loss_config: DictConfig = None,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval

        out_channels = out_channels if out_channels is not None else in_channels

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

        self.dim = dim
        self.hidden_dims = hidden_dims
        self.num_groups = num_groups
        self.act = act

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                conv_nd(
                    dim,
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

        in_ch = embed_dim
        skip_dims = []

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                self.encoder.append(
                    SDBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )

                in_ch = out_ch
                skip_dims.append(in_ch)

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        dim=2,
                        pool_type=pool_type,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )

        self.middle = nn.Sequential(
            SDBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                drop_path=drop_path,
                deformable_groups=deformable_groups,
                deformable_group_channels=deformable_group_channels,
                offset_scale=offset_scale,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size,
                act=act,
                use_conv=use_conv,
                num_groups=num_groups,
                dim=dim,
                use_checkpoint=use_checkpoint
            ),
            DeformableResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                drop_path=drop_path,
                act=act,
                dim=dim,
                deformable_groups=deformable_groups,
                deformable_group_channels=deformable_group_channels,
                offset_scale=offset_scale,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size,
                num_groups=num_groups,
                use_checkpoint=use_checkpoint,
                use_conv=use_conv,
            )
        )

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                in_ch = in_ch + skip_dims.pop()
                self.decoder.append(
                    SDBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )

                in_ch = out_ch

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        mode=mode,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )

        self.out = nn.Sequential(
            group_norm(in_ch, num_groups=num_groups),
            get_act(act),
            deform_conv_nd(
                dim,
                in_ch,
                out_channels,
                kernel_size=3,
                padding=1,
                deformable_groups_per_groups=in_ch,
                offset_scale=offset_scale,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size,
            ),
            nn.Sigmoid(),
        )

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            if not isinstance(block, DownBlock):
                hs.append(h)

        h = self.middle(h)

        for i, block in enumerate(self.decoder):
            if not isinstance(block, UpBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        return self.out(h)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        net_loss, net_loss_log = self.loss(preds, labels, imgs, split='train')

        if self.global_step % self.log_interval == 0:
            self.log_img(preds, 'edge')
            self.log_img(labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(net_loss_log, prog_bar=True)

        return net_loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        net_loss, net_loss_log = self.loss(preds, labels, imgs, split='val')

        if self.global_step % self.log_interval == 0:
            self.log_img(preds, 'edge')
            self.log_img(labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(net_loss_log,  prog_bar=True)

    @torch.no_grad()
    def log_img(self, x: torch.Tensor, split='img'):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/{split}', x[0].float(), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        params = list(self.parameters())
        opt_net = torch.optim.AdamW(params,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        opts = [opt_net]

        return opts
