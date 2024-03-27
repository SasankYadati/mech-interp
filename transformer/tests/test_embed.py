from transformer.Embed import Embed
from transformer.Config import Config
import torch as t
from transformer_lens import HookedTransformer

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)

class TestEmbed():
    def test_shape(self):
        cfg = Config()
        l = Embed(cfg)
        x = t.randint(0, cfg.d_vocab, size=(32, 8))
        y = l(x)
        assert y.shape == (32, 8, cfg.d_model)

    def test_with_reference_gpt2(self):
        cfg = Config(debug=True)
        l = Embed(cfg)
        l.load_state_dict(reference_gpt2.embed.state_dict())
        x = t.randint(0, cfg.d_vocab, size=(32, 8))
        y_out = l(x)
        y_expected = reference_gpt2.embed(x)
        comparison = t.isclose(y_out, y_expected, atol=1e-4, rtol=1e-3)
        assert comparison.sum()/comparison.numel() >= 0.99999