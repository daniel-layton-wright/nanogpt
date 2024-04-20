from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    embed_dim: int = 64
    num_heads: int = 4
    context_size: int = 128
    dropout: float = 0.2
    vocab_size: int = 10000
    num_blocks: int = 4


class AttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        self.dropout = nn.Dropout(config.dropout)

        mask = ~(torch.tril(torch.ones(config.context_size, config.context_size)).view(
            1, 1, config.context_size, config.context_size
        ).to(torch.bool))

        self.register_buffer('mask', mask)

    def forward(self, x):
        h = self.config.num_heads
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Currently q is (B, T, C)
        # if we change it to (B, T, H, D) this is the per-head version
        # H is the number of heads. D is the dimension of each head
        # Then we can reshape to (B, H, T, D)
        q = q.view(B, T, h, -1).transpose(1, 2)
        k = k.view(B, T, h, -1).transpose(1, 2)
        v = v.view(B, T, h, -1).transpose(1, 2)

        # Now the q, k, v are ready to be masked
        # B H T D    x    B H D T  ->    B H T T
        a = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

        # Need to mask
        a = a.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        a = a.softmax(dim=-1)
        a = self.dropout(a)

        # Weighted sums of values
        out = a @ v  # B H T T x B H T D -> B H T D

        # Reshape back to B T C
        out = out.transpose(1, 2).reshape(B, T, -1)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = AttentionBlock(config)
        self.resid_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.ffwd = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.resid_dropout(self.resid_proj(self.attn(self.ln1(x))))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embeddings = nn.Embedding(config.context_size, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_blocks)])
        self.transformer_ln = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embeddings(x) + self.pos_embeddings(torch.arange(T, device=x.device))
        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.transformer_ln(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_new_tokens: int, temperature: float):
        """
        Sample tokens from the model

        :param tokens:
        :param n_new_tokens:
        :param temperature:
        :return:
        """

        for _ in range(n_new_tokens):
            tokens = tokens[:, -self.config.context_size:]
            logits = self(tokens)
            logits = logits[:, -1] / temperature
            probs = logits.softmax(dim=-1)
            sampled_token_idx = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, sampled_token_idx], dim=-1)

        return tokens
