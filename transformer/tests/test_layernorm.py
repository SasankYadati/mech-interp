from transformer.LayerNorm import LayerNorm
from transformer.Config import Config
import torch as t

class TestLayerNorm():
    def test_shape(self):
        d_model = 4
        cfg = Config(layer_norm_eps=1e-5, d_model=d_model)
        l = LayerNorm(cfg)
        x = t.randn((32, 3, d_model))
        y = l(x)

        assert x.shape == y.shape