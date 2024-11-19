import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDropout(nn.Module):
    def __init__(self,
                 p: float = 0.5,
                 inplace: bool = False,
                 *args,
                 ):
        super().__init__()

        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0. or not self.training:
            return x

        keep_prob = 1 - self.p
        keep = torch.empty(x.shape, device=x.device).bernoulli_(keep_prob)
        if keep_prob > 0:
            keep = keep / keep_prob
        return x * keep


class ComplexDropPath(nn.Module):
    def __init__(self,
                 p: float = 0.,
                 scale_by_keep: bool = True,
                 ):
        super().__init__()
        self.p = p
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x

        keep_prob = 1 - self.p
        random_tensor = torch.empty(x.shape).bernoulli_(keep_prob).to(device=x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
