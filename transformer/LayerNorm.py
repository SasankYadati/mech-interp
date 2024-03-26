import torch as t
from torch import nn
from torch import Tensor
from transformer.Config import Config
from jaxtyping import Float, Int

class LayerNorm(nn.Module):
    '''
    Refer to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    for implementation.
    We will always assume elementwise_affine and bias.
    '''
    def __init__(self, cfg:Config):
        super().__init__()
        self.eps = cfg.layer_norm_eps
        self.gamma = nn.Parameter(t.ones(cfg.d_model))
        self.beta = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        mean = residual.mean(dim=0, keepdim=True)
        var = residual.std(dim=0, keepdim=True) ** 2
        out = (residual - mean) / (var + self.eps) ** (0.5)
        out = out * self.gamma + self.beta
        return out
