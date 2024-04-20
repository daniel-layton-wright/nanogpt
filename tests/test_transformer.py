import unittest
import torch


class TestTransformer(unittest.TestCase):
    def test_attention_block(self):
        from nanogpt.transformer import AttentionBlock, TransformerConfig

        config = TransformerConfig()
        attn = AttentionBlock(config)

        x = torch.randn(100, config.context_size, config.embed_dim)
        y = attn(x)

        # Check size is correct
        self.assertEqual(y.shape, x.shape)

    def test_transformer(self):
        from nanogpt.transformer import Transformer, TransformerConfig

        config = TransformerConfig()
        transformer = Transformer(config)

        x = torch.randint(0, config.vocab_size, (100, config.context_size))
        y = transformer(x)

        # Check size is correct
        self.assertEqual(y.shape, (100, config.context_size, config.vocab_size))

    def test_generation(self):
        from nanogpt.transformer import Transformer, TransformerConfig

        config = TransformerConfig()
        transformer = Transformer(config)

        x = torch.tensor([1], dtype=torch.long).reshape(1, 1)
        y = transformer.generate(x, 10, 1.0)

        self.assertEqual(y.shape, (1, 11))
