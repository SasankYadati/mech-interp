import torch as t
from torch import nn
from torch import Tensor
from transformer.config import Config
from transformer.layer_norm import LayerNorm
from transformer.attention import Attention
from transformer.mlp import MLP
from jaxtyping import Float, Int

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        x = self.attn(self.ln1(resid_pre)) + resid_pre
        return x + self.mlp(self.ln2(x))