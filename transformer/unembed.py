import torch as t
from torch import nn
from torch import Tensor
from transformer.config import Config
from jaxtyping import Float, Int
import einops

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_vocab"]:
        out = einops.einsum(
            normalized_resid_final, self.W_U, 
            "batch pos d_model, d_model d_vocab -> batch pos d_vocab"
        ) + self.b_U
        return out
