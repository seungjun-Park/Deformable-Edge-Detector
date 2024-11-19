import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight, freq_mask
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss, wasserstein_d_loss, LFD
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 cats_weight: List[float] = (1.0, 0.0, 1.2),
                 lpips_weight: float = 1.0,
                 balanced_l1_weight: float = 1.0,
                 bdcn_weight: float = 1.0,
                 content_weight: float = 0.5,
                 ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.lpips_weight = lpips_weight
        self.balanced_l1_weight = balanced_l1_weight
        self.bdcn_weight = bdcn_weight
        self.cats_weight = cats_weight
        self.content_weight = content_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split="train"):
        balanced_l1_loss = (torch.abs(preds - labels)).mean()

        preds = preds.repeat(1, 3, 1, 1).contiguous()
        labels = labels.repeat(1, 3, 1, 1).contiguous()

        lpips_loss = self.perceptual_loss(preds, labels).mean()
        content_loss = self.perceptual_loss(preds, imgs).mean()

        # loss = self.lpips_weight * lpips_loss + cats
        loss = self.lpips_weight * lpips_loss + self.balanced_l1_weight * balanced_l1_loss + self.content_weight * content_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/lpips_loss".format(split): lpips_loss.detach().mean(),
               "{}/l1_loss".format(split): balanced_l1_loss.detach().mean(),
               "{}/content_loss".format(split): content_loss.detach().mean(),
               }

        return loss, log






