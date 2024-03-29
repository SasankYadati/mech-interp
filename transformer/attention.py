import torch as t
from torch import nn
from torch import Tensor
from transformer.config import Config
from jaxtyping import Float, Int
import einops

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class Attention(nn.Module):
    EPSILON: Float[Tensor, ""]
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n, e, h = cfg.n_heads, cfg.d_model, cfg.d_head
        
        self.W_Q = nn.Parameter(t.empty((n, e, h)))
        self.b_Q = nn.Parameter(t.zeros((n, h)))
        
        self.W_K = nn.Parameter(t.empty((n, e, h)))
        self.b_K = nn.Parameter(t.zeros((n, h)))
        
        self.W_V = nn.Parameter(t.empty((n, e, h)))
        self.b_V = nn.Parameter(t.zeros((n, h)))

        self.W_O = nn.Parameter(t.empty(n, h, e))
        self.b_O = nn.Parameter(t.zeros(e))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)

        self.register_buffer("EPSILON", t.tensor(-1e5, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        # b - num batches, s - seq len, e - d_model, n - n_heads, h - d_heads

        keys = einops.einsum(normalized_resid_pre, self.W_K, "b s e, n e h -> b s n h") + self.b_K
        queries = einops.einsum(normalized_resid_pre, self.W_Q, "b s e, n e h -> b s n h") + self.b_Q
        
        attn_scores = einops.einsum(queries, keys, "b s1 n h, b s2 n h -> b n s1 s2")
        attn_scores /= self.cfg.d_head ** 0.5
        attn_scores = self.apply_causal_mask(attn_scores)
        
        attn_probs = attn_scores.softmax(dim=-1)

        values = einops.einsum(normalized_resid_pre, self.W_V, "b s e, n e h -> b s n h") + self.b_V
        
        z = einops.einsum(attn_probs, values, "b n s1 s2, b s2 n h -> b s1 n h")

        result = einops.einsum(z, self.W_O, "b s n h, n h e -> b s n e") 

        attn_out = einops.reduce(result, "b s n e -> b s e", 'sum') + self.b_O

        return attn_out


    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        sq, sk = attn_scores.shape[-2], attn_scores.shape[-1]
        ones = t.ones(sq, sk, device=attn_scores.device)
        mask = t.triu(ones, diagonal=1).bool() 
        attn_scores.masked_fill_(mask, self.EPSILON)
        return attn_scores