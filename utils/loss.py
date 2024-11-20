import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Dict
from torch.fft import rfftn, ifftn

from utils import conv_nd


# Frequency Distance loss
def FD(target, pred, dim=2, type='l2'):
    type = type.lower()
    assert type in ['l1', 'l2']
    dim = [-i for i in range(dim, 0, -1)]
    target_freq = rfftn(target, dim=dim)
    pred_freq = rfftn(pred, dim=dim)

    # general form of frequency distance
    fd = (target_freq - pred_freq).abs()
    if type == 'l2':
        fd = torch.square(fd)
    fd = torch.mean(fd, dim=[-2, -1])
    fd = torch.sum(fd, dim=[-1])
    fd = torch.sum(fd) / fd.shape[0]
    return fd


# Log Frequency Distance loss
def LFD(target, pred, dim=2, type='l1'):
    fd = FD(target, pred, dim=dim, type=type)
    lfd = torch.log(fd + 1)

    return lfd


class CosineDistance(nn.Module):
    def __init__(self,
                 dim: int = 1,
                 deep_supervision_dim: int = None,
                 ):
        super().__init__()
        self.dim = dim
        self.deep_supervision_dim = deep_supervision_dim

        if deep_supervision_dim is not None:
            self.weight = nn.Parameter(torch.ones((1, deep_supervision_dim)), requires_grad=True)

    def forward(self,
                inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
                targets: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
                ) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        cos_dists = list()

        for ip, tg in zip(inputs, targets):
            cos_dist = 1.0 - F.cosine_similarity(ip, tg, dim=self.dim)
            cos_dist = torch.mean(cos_dist, dim=[i for i in range(1, cos_dist.ndim)]).unsqueeze(1)
            cos_dists.append(cos_dist)

        if self.deep_supervision_dim is not None:
            cos_dists = torch.cat(cos_dists, dim=1)
            cost = F.linear(cos_dists, weight=torch.clamp(self.weight, min=0.0, max=1.0))

        else:
            cost = cos_dists[0]

        cost = torch.mean(cost)

        return cost


class EuclideanDistance(nn.Module):
    def __init__(self,
                 use_square: bool = False,
                 reduction_dim: Union[int, List[int], Tuple[int]] = (2, 3),
                 reduction: str = 'sum',
                 in_channels: int = None,
                 use_weight: bool = True
                 ):
        super().__init__()

        self.use_square = use_square
        self.reduction_dim = tuple(reduction_dim)
        reduction = reduction.lower()
        assert reduction in ['sum', 'mean']
        self.reduction = reduction.lower()
        self.use_weight = use_weight

        if use_weight:
            assert in_channels is not None
            self.weight = nn.Linear(in_features=in_channels, out_features=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        euclidean_dist = torch.pow(inputs - targets, 2)  # euclidean_dist.shape == [B, *]

        if self.use_weight:
            euclidean_dist = torch.flatten(euclidean_dist, start_dim=1)
            b, n = euclidean_dist.shape
            euclidean_dist = self.weight(euclidean_dist)
            euclidean_dist = F.relu(euclidean_dist)
            if self.reduction == 'mean':
                euclidean_dist = euclidean_dist / n
            euclidean_dist = torch.sum(euclidean_dist) / b

        else:
            if self.reduction == 'sum':
                euclidean_dist = torch.sum(euclidean_dist, dim=self.reduction_dim)
                euclidean_dist = torch.sum(euclidean_dist) / euclidean_dist.shape[0]
            elif self.reduction == 'mean':
                euclidean_dist = torch.mean(euclidean_dist, dim=self.reduction_dim)
                euclidean_dist = torch.sum(euclidean_dist) / euclidean_dist.shape[0]

            if not self.use_square:
                euclidean_dist = torch.sqrt(euclidean_dist)

        return euclidean_dist

    def normalize_tensor(self, x, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)


class CosineSimilarity(nn.Module):
    def __init__(self,
                 dim: int = 1,
                 reduction: str = 'mean',
                 ):
        super().__init__()

        self.dim = dim
        self.reduction = reduction.lower()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_dist = cosine_distance(inputs,
                                   target,
                                   dim=self.dim,
                                   reduction=self.reduction)


        return  cos_dist


class EuclideanDistanceWithCosineDistance(nn.Module):
    def __init__(self,
                 use_square: bool = False,
                 use_normalize: bool = False,
                 ed_weight: float = 1.0,
                 ed_dim: Union[int, List[int], Tuple[int]] = (2, 3),
                 cd_weight: float = 1.0,
                 cd_dim: int = 1,
                 reduction: str = 'none',
                 ):
        super().__init__()

        self.use_square = use_square
        self.use_normalize = use_normalize
        self.ed_weight = ed_weight
        self.ed_dim = tuple(ed_dim)
        self.cd_weight = cd_weight
        self.cd_dim = cd_dim
        self.reduction = reduction.lower()

    def forward(self,
                inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
                targets: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        euclidean_dists = 0
        cos_dists = 0

        for ips, tgs in zip(inputs, targets):
            if self.use_normalize:
                euclidean_dist = normalized_euclidean_distance(ips,
                                                               tgs,
                                                               dim=self.ed_dim,
                                                               reduction=self.reduction,
                                                               use_square=self.use_square)

            else:
                euclidean_dist = euclidean_distance(ips, tgs, reduction=self.reduction, use_square=self.use_square)

            euclidean_dists = euclidean_dists + euclidean_dist

            cos_dist = cosine_distance(ips,
                                       tgs,
                                       dim=self.cd_dim,
                                       reduction=self.reduction)

            cos_dists = cos_dists + cos_dist

        return euclidean_dists * self.ed_weight + cos_dists * self.cd_weight


def jaccard_distance(inputs: torch.Tensor, targets: torch.Tensor):
    return


def euclidean_distance(inputs: torch.Tensor, targets: torch.Tensor,
                       reduction: str = 'mean', use_square: bool = False):
    euclidean_dist = F.mse_loss(inputs, targets, reduction=reduction)
    if not use_square:
        return torch.sqrt(euclidean_dist)

    return euclidean_dist


def normalized_euclidean_distance(inputs: torch.Tensor, targets: torch.Tensor, dim: Union[int, List[int], Tuple[int]] = 1,
                                  eps: float = 1e-5, reduction: str = 'mean', use_square: bool = False
                                  ) -> torch.Tensor:
    reduction = reduction.lower()

    std_dist = 0.5 * (torch.std(inputs - targets, dim=dim, keepdim=True) /
                      (torch.std(inputs, dim=dim, keepdim=True) + torch.std(targets, dim=dim, keepdim=True) + eps))

    if reduction == 'mean':
        std_dist = torch.mean(std_dist)
    elif reduction == 'sum':
        std_dist = torch.sum(std_dist) / std_dist.shape[0]
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError(f'reduction: "{reduction}" is not implemented.')

    if use_square:
        return std_dist

    return torch.sqrt(std_dist)


def cosine_distance(inputs: torch.Tensor, targets: torch.Tensor, dim: int = 1,
                    reduction: str = 'mean',
                    ) -> torch.Tensor:
    cos_dist = 1.0 - F.cosine_similarity(inputs, targets, dim=dim)

    reduction = reduction.lower()

    if reduction == 'mean':
        cos_dist = torch.mean(cos_dist)
    elif reduction == 'sum':
        cos_dist = torch.sum(cos_dist) / cos_dist.shape[0]
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError(f'reduction: "{reduction}" is not implemented.')

    return cos_dist


def san_d_loss(out_real: Dict[str, torch.Tensor], out_fake: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits_real = out_real['logits']
    logits_fake = out_fake['logits']
    dir_real = out_real['dir']
    dir_fake = out_fake['dir']

    loss_hinge = hinge_d_loss(logits_real, logits_fake)
    loss_wasserstein = wasserstein_d_loss(dir_real, dir_fake)
    return loss_hinge + loss_wasserstein


def wasserstein_d_loss(dir_real: torch.Tensor, dir_fake: torch.Tensor) -> torch.Tensor:
    return dir_fake.mean() - dir_real.mean()


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def bdcn_loss2(inputs, targets):
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost)

    return cost


# ------------ cats losses ----------


def bdrloss(prediction, label, radius):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1).to(prediction.device)
    filt.requires_grad = False

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    texture_mask = F.conv2d(label, filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()


def textureloss(prediction, label, mask_radius):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3).to(prediction.device)
    filt1.requires_grad = False
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1).to(prediction.device)
    filt2.requires_grad = False

    pred_sums = F.conv2d(prediction, filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label, filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).to(prediction.dtype)

    loss = torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10)
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, weights=(1., 0., 0.)):
    # tracingLoss
    cost_weight, tex_factor, bdr_factor = weights
    balanced_w = 1.1

    label = label.long().float()
    prediction = prediction.float()

    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(prediction, label, weight=mask, reduction='none')
    cost = torch.sum(cost)
    label_w = (label != 0).float()

    textcost = textureloss(prediction, label_w, mask_radius=4)
    bdrcost = bdrloss(prediction, label_w, radius=4)

    return cost_weight * cost + bdr_factor * bdrcost + tex_factor * textcost, cost, bdrcost, textcost


SHIFT = torch.Tensor([-.030, -.088, -.188])[None, :, None, None]
SCALE = torch.Tensor([.458, .448, .450])[None, :, None, None]


def scaling(x: torch.Tensor):
    shift = SHIFT.to(x.device)
    scale = SCALE.to(x.device)
    return (x - shift) / scale


def contents_loss(net, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    in0_input, in1_input = (scaling(inputs), scaling(targets))
    feats0, feats1 = net(in0_input), net(in1_input)
    diffs = []

    for kk in range(5):
        diffs.append(torch.mean((feats0[kk] - feats1[kk]) ** 2, dim=[2, 3]))

    val = torch.cat(diffs, dim=1)
    val = torch.sum(val, dim=1)
    val = torch.sum(val) / val.shape[0]

    return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale