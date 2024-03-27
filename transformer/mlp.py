import torch as t
from torch import nn
from torch import Tensor
from transformer_lens.utils import gelu_new
from transformer.config import Config
from jaxtyping import Float, Int
import einops

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        out = einops.einsum(
            normalized_resid_mid, self.W_in, 
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_in
        out = gelu_new(out)
        out = einops.einsum(
            out, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out
        return out

