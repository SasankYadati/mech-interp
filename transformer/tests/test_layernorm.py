from transformer.layer_norm import LayerNorm
from transformer.config import Config
import torch as t
from transformer_lens import HookedTransformer

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)

class TestLayerNorm():
    def test_shape(self):
        d_model = 4
        cfg = Config(layer_norm_eps=1e-5, d_model=d_model)
        l = LayerNorm(cfg)
        x = t.randn((32, 3, d_model))
        y = l(x)

        assert x.shape == y.shape

    def test_with_reference_gpt2(self):
        cfg = Config(debug=True)
        l = LayerNorm(cfg)
        l.load_state_dict(reference_gpt2.ln_final.state_dict())
        x = t.randn((32, 8, cfg.d_model))
        y_out = l(x)
        y_expected = reference_gpt2.ln_final(x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.99999