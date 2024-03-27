import torch as t
from torch import nn
from torch import Tensor
from transformer.config import Config
from jaxtyping import Float, Int

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)
    
    def forward(self, tokens:Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        b, p = tokens.shape
        positions = t.arange(0, p).repeat((b, 1))
        return self.W_pos[positions]