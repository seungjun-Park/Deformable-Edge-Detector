import torch
import torch.nn as nn
import torch.nn.functional as F


from .activation import Sine, Cosine


def functional_conv_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return F.conv1d(*args, **kwargs)

    elif dim == 2:
        return F.conv2d(*args, **kwargs)

    elif dim == 3:
        return F.conv3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def conv_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return nn.Conv1d(*args, **kwargs)

    elif dim == 2:
        return nn.Conv2d(*args, **kwargs)

    elif dim == 3:
        return nn.Conv3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def conv_transpose_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return nn.ConvTranspose1d(*args, **kwargs)

    elif dim == 2:
        return nn.ConvTranspose2d(*args, **kwargs)

    elif dim == 3:
        return nn.ConvTranspose3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def avg_pool_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return nn.AvgPool1d(*args, **kwargs)

    elif dim == 2:
        return nn.AvgPool2d(*args, **kwargs)

    elif dim == 3:
        return nn.AvgPool3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def max_pool_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return nn.MaxPool1d(*args, **kwargs)

    elif dim == 2:
        return nn.MaxPool2d(*args, **kwargs)

    elif dim == 3:
        return nn.MaxPool3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def pool_nd(pool_type: str = 'max', dim: int = 2, *args, **kwargs):
    pool_type = pool_type.lower()
    if pool_type == 'max':
        return max_pool_nd(dim=dim, *args, **kwargs)
    elif pool_type in ['avg', 'average']:
        return avg_pool_nd(dim=dim, *args, **kwargs)
    else:
        NotImplementedError('current version should be supported max and avg pooling')


def batch_norm_nd(dim=2, *args, **kwargs):
    if dim == 1:
        return nn.BatchNorm1d(*args, **kwargs)

    elif dim == 2:
        return nn.BatchNorm2d(*args, **kwargs)

    elif dim == 3:
        return nn.BatchNorm3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def instance_norm_nd(dim=2, *args, **kwargs):
    if dim == 1:
        return nn.InstanceNorm1d(*args, **kwargs)

    elif dim == 2:
        return nn.InstanceNorm2d(*args, **kwargs)

    elif dim == 3:
        return nn.InstanceNorm3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")


def group_norm(num_channels, num_groups=32, eps=1e-6, affine=True):
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


# to allow mixed-precision training
class CustomGroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def norm(norm_type='group', groups=32, *args, **kwargs):
    norm_type = norm_type.lower()
    if norm_type == 'group':
        return group_norm(num_groups=groups, *args, **kwargs)

    elif norm_type == 'layer':
        return nn.LayerNorm(*args, **kwargs)

    elif norm_type == 'instance':
        return instance_norm_nd(*args, **kwargs)

    elif norm_type == 'batch':
        return batch_norm_nd(*args, **kwargs)

    else:
        NotImplementedError(f"The norm_type: {norm_type} is not supported.")


def get_act(name='relu', *args, **kwargs):
    name = name.lower()

    if name == 'relu':
        return nn.ReLU(*args, **kwargs)

    elif name == 'softplus':
        return nn.Softplus(*args, **kwargs)

    elif name == 'silu':
        return nn.SiLU(*args, **kwargs)

    elif name == 'sigmoid':
        return nn.Sigmoid(*args, **kwargs)

    elif name == 'tanh':
        return nn.Tanh(*args, **kwargs)

    elif name == 'hard_tanh':
        return nn.Hardtanh(*args, **kwargs)

    elif name == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)

    elif name == 'elu':
        return nn.ELU(*args, **kwargs)

    elif name == 'gelu':
        return nn.GELU(*args, **kwargs)

    elif name == 'mish':
        return nn.Mish(*args, **kwargs)

    elif name == 'sine':
        return Sine(*args, **kwargs)

    elif name == 'cosine':
        return Cosine(*args, **kwargs)

    else:
        NotImplementedError(f'Activation function "{name}" is not supported.')