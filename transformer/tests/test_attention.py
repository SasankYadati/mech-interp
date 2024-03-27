from transformer.attention import Attention
from transformer.config import Config
import torch as t
from transformer_lens import HookedTransformer

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)

class TestAttention():
    def test_shape(self):
        cfg = Config()
        l = Attention(cfg)
        x = t.randn((32, 16, cfg.d_model))
        y = l(x)

        assert x.shape == y.shape

    def test_with_reference_gpt2(self):
        cfg = Config()
        l = Attention(cfg)
        l.load_state_dict(reference_gpt2.blocks[0].attn.state_dict(), strict=False)
        x = t.randn((32, 16, cfg.d_model))
        y_out = l(x)
        y_expected = reference_gpt2.blocks[0].attn(x, x, x) # self attention
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.999