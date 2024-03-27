from transformer.mlp import MLP
from transformer.config import Config
import torch as t
from transformer.tests import reference_gpt2

class TestEmbed():
    def test_shape(self):
        cfg = Config()
        l = MLP(cfg)
        x = t.randn(32, 8, cfg.d_model)
        y = l(x)
        assert y.shape == (32, 8, cfg.d_model)

    def test_with_reference_gpt2(self):
        cfg = Config(debug=True)
        l = MLP(cfg)
        l.load_state_dict(reference_gpt2.blocks[0].mlp.state_dict())
        x = t.randn(32, 8, cfg.d_model)
        y_out = l(x)
        y_expected = reference_gpt2.blocks[0].mlp(x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.99999