import torch as t
from torch import nn
from torch import Tensor
from transformer.config import Config
from jaxtyping import Float, Int

class LayerNorm(nn.Module):
    '''
    Refer to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    for implementation.
    We will assume elementwise_affine and bias.
    We will calculate the mean and variance over the last dimension.
    '''
    def __init__(self, cfg:Config):
        super().__init__()
        self.eps = cfg.layer_norm_eps
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        mean = t.mean(residual, dim=-1, keepdim=True)
        var = t.var(residual, dim=-1, keepdim=True, unbiased=False)
        out = (residual - mean) / ((var + self.eps) ** (0.5))
        out = out * self.w + self.b
        return out
