from transformer.unembed import Unembed
from transformer.config import Config
import torch as t
from transformer.tests import reference_gpt2

class TestEmbed():
    def test_shape(self):
        cfg = Config()
        l = Unembed(cfg)
        x = t.randn((32, 16, cfg.d_model))
        y = l(x)
        assert y.shape == (32, 16, cfg.d_vocab)

    def test_with_reference_gpt2(self):
        cfg = Config(debug=True)
        l = Unembed(cfg)
        l.load_state_dict(reference_gpt2.unembed.state_dict())
        x = t.randn((32, 16, cfg.d_model))
        y_out = l(x)
        y_expected = reference_gpt2.unembed(x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.99999