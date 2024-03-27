from transformer.transformer_block import TransformerBlock
from transformer.config import Config
import torch as t
from transformer.tests import reference_gpt2

class TestTransformerBlock:
    def test_shape(self):
        cfg = Config()
        l = TransformerBlock(cfg)
        x = t.randn((32, 16, cfg.d_model))
        y = l(x)

        assert x.shape == y.shape

    def test_with_reference_gpt2(self):
        cfg = Config()
        l = TransformerBlock(cfg)
        l.load_state_dict(reference_gpt2.blocks[0].state_dict(), strict=False)
        x = t.randn((32, 16, cfg.d_model))
        y_out = l(x)
        y_expected = reference_gpt2.blocks[0](x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.999