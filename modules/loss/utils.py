from torchvision.models.swin_transformer import (
    swin_t, swin_s, swin_b,  swin_v2_t, swin_v2_s, swin_v2_b,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
)
from torchvision.models.vision_transformer import (
    vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_H_14_Weights, ViT_L_16_Weights, ViT_L_32_Weights
)
from torchvision.models.vgg import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    VGG11_Weights, VGG11_BN_Weights, VGG13_Weights, VGG13_BN_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights
)
from torchvision.models.alexnet import (
    alexnet,
    AlexNet_Weights
)
from torchvision.models.resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)
from torchvision.models.squeezenet import (
    squeezenet1_0, squeezenet1_1,
    SqueezeNet1_0_Weights, SqueezeNet1_1_Weights
)


import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pretrained_model(net_type='alexnet', device='cpu'):
    net_type = net_type.lower()
    net = None
    if net_type == 'alexnet':
        net = alexnet(AlexNet_Weights.DEFAULT)

    elif net_type == 'resnet18':
        net = resnet18(ResNet18_Weights.DEFAULT)
    elif net_type == 'resnet34':
        net = resnet34(ResNet34_Weights.DEFAULT)
    elif net_type == 'resnet50':
        net = resnet50(ResNet50_Weights.DEFAULT)
    elif net_type == 'resnet101':
        net = resnet101(ResNet101_Weights.DEFAULT)
    elif net_type == 'resnet152':
        net = resnet152(ResNet152_Weights.DEFAULT)

    elif net_type == 'vgg11':
        net = vgg11(VGG11_Weights.DEFAULT)
    elif net_type == 'vgg11_bn':
        net = vgg11_bn(VGG11_BN_Weights.DEFAULT)
    elif net_type == 'vgg13':
        net = vgg13(VGG13_Weights.DEFAULT)
    elif net_type == 'vgg13_bn':
        net = vgg13_bn(VGG13_BN_Weights.DEFAULT)
    elif net_type == 'vgg16':
        net = vgg16(VGG16_Weights.DEFAULT)
    elif net_type == 'vgg16_bn':
        net = vgg16_bn(VGG16_BN_Weights.DEFAULT)
    elif net_type == 'vgg19':
        net = vgg19(VGG19_Weights.DEFAULT)
    elif net_type == 'vgg19_bn':
        net = vgg19_bn(VGG11_Weights.DEFAULT)

    elif net_type == 'vit_b_16':
        net = vit_b_16(ViT_B_16_Weights.DEFAULT)
    elif net_type == 'vit_b_32':
        net = vit_b_32(ViT_B_32_Weights.DEFAULT)
    elif net_type == 'vit_h_14':
        net = vit_h_14(ViT_H_14_Weights.DEFAULT)
    elif net_type == 'vit_l_16':
        net = vit_l_16(ViT_L_16_Weights.DEFAULT)
    elif net_type == 'vit_l_32':
        net = vit_l_32(ViT_L_32_Weights.DEFAULT)

    elif net_type == 'swin_t':
        net = swin_t(Swin_T_Weights.DEFAULT)
    elif net_type == 'swin_s':
        net = swin_s(Swin_S_Weights.DEFAULT)
    elif net_type == 'swin_s':
        net = swin_b(Swin_B_Weights.DEFAULT)
    elif net_type == 'swin_v2_t':
        net = swin_v2_t(Swin_V2_T_Weights.DEFAULT)
    elif net_type == 'swin_v2_s':
        net = swin_v2_s(Swin_V2_S_Weights.DEFAULT)
    elif net_type == 'swin_v2_b':
        net = swin_v2_b(Swin_V2_B_Weights.DEFAULT)

    else:
        NotImplementedError(f'{net_type} is not implemented.')

    return net.to(device=device)


def export_layers(net, net_type: str):
    net_type = net_type.lower()
    if '_' in net_type:
        net_type = net_type.rsplit('_')

    layers = []

    if net_type == 'alexnet':
        net = alexnet(AlexNet_Weights.DEFAULT)

    elif net_type == 'resnet18':
        net = resnet18(ResNet18_Weights.DEFAULT)
    elif net_type == 'resnet34':
        net = resnet34(ResNet34_Weights.DEFAULT)
    elif net_type == 'resnet50':
        net = resnet50(ResNet50_Weights.DEFAULT)
    elif net_type == 'resnet101':
        net = resnet101(ResNet101_Weights.DEFAULT)
    elif net_type == 'resnet152':
        net = resnet152(ResNet152_Weights.DEFAULT)

    elif net_type == 'vgg11':
        net = vgg11(VGG11_Weights.DEFAULT)
    elif net_type == 'vgg11_bn':
        net = vgg11_bn(VGG11_BN_Weights.DEFAULT)
    elif net_type == 'vgg13':
        net = vgg13(VGG13_Weights.DEFAULT)
    elif net_type == 'vgg13_bn':
        net = vgg13_bn(VGG13_BN_Weights.DEFAULT)
    elif net_type == 'vgg16':
        net = vgg16(VGG16_Weights.DEFAULT)
    elif net_type == 'vgg16_bn':
        net = vgg16_bn(VGG16_BN_Weights.DEFAULT)
    elif net_type == 'vgg19':
        net = vgg19(VGG19_Weights.DEFAULT)
    elif net_type == 'vgg19_bn':
        net = vgg19_bn(VGG11_Weights.DEFAULT)

    elif net_type == 'vit_b_16':
        net = vit_b_16(ViT_B_16_Weights.DEFAULT)
    elif net_type == 'vit_b_32':
        net = vit_b_32(ViT_B_32_Weights.DEFAULT)
    elif net_type == 'vit_h_14':
        net = vit_h_14(ViT_H_14_Weights.DEFAULT)
    elif net_type == 'vit_l_16':
        net = vit_l_16(ViT_L_16_Weights.DEFAULT)
    elif net_type == 'vit_l_32':
        net = vit_l_32(ViT_L_32_Weights.DEFAULT)

    elif 'swin' in net_type:
        dims = (0, 3, 1, 2)
        reverse_dims = (0, 2, 3, 1)
        layers.append(nn.Sequential(net.features[:2], Permute(dims=dims)))
        layers.append(nn.Sequential(Permute(dims=reverse_dims), net.features[2: 4], Permute(dims=dims)))
        layers.append(nn.Sequential(Permute(dims=reverse_dims), net.features[4: 6], Permute(dims=dims)))
        layers.append(nn.Sequential(Permute(dims=reverse_dims), net.features[6: ], Permute(dims=dims)))

    else:
        NotImplementedError(f'{net_type} is not implemented.')

    return layers


def get_layer_dims(net_type):
    if '_' in net_type:
        net_type = net_type.rsplit('_')

    if 'vit' in net_type:
        return

    elif 'swin' in net_type:
        if 't' in net_type[-1]:
            embed_dim = 96
            dims = [embed_dim * 2 ** i for i in range(4)]
        elif 's' in net_type[-1]:
            embed_dim = 96
            dims = [embed_dim * 2 ** i for i in range(4)]
        elif 'b' in net_type[-1]:
            embed_dim = 128
            dims = [embed_dim * 2 ** i for i in range(4)]

    else:
        NotImplementedError(f'{net_type} is not implemented.')
    return dims


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([-2, -1], keepdim=keepdim)


class Permute(nn.Module):
    def __init__(self,
                 dims,
                 *args,
                 **kwargs):
        super().__init__()

        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
