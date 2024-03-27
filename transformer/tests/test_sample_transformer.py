from transformer.sample_transformer import SampleTransformer
from transformer.config import Config
import torch as t
from transformer.tests import reference_gpt2

class TestSampleTransformer:
    def test_shape(self):
        cfg = Config()
        l = SampleTransformer(cfg)
        x = t.randint(0, cfg.d_vocab, size=(32, 8))
        y = l(x)

        assert y.shape == (32, 8, cfg.d_vocab)

    def test_with_reference_gpt2(self):
        cfg = Config()
        l = SampleTransformer(cfg)
        l.load_state_dict(reference_gpt2.state_dict(), strict=False)
        x = t.randint(0, cfg.d_vocab, size=(32, 8))
        y_out = l(x)
        y_expected = reference_gpt2(x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.999