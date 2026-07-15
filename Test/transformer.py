from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_token_id: int = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(dtype=x.dtype)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = d_model**0.5

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens) * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim**0.5)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(padding_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = attn @ v
        return self.out_proj(self._merge_heads(context))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x),
                self.norm1(x),
                self.norm1(x),
                key_padding_mask=src_key_padding_mask,
            )
        )
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
        )

        x = x + self.dropout(
            self.cross_attn(
                self.norm2(x),
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
            )
        )
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.src_embedding = TokenEmbedding(config.src_vocab_size, config.d_model)
        self.tgt_embedding = TokenEmbedding(config.tgt_vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
        )

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens.eq(self.config.pad_token_id)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode(self, src_tokens: Tensor) -> Tensor:
        src_padding_mask = self.make_padding_mask(src_tokens)
        x = self.src_embedding(src_tokens)
        x = self.dropout(self.positional_encoding(x))
        return self.encoder(x, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt_tokens: Tensor, memory: Tensor, src_tokens: Tensor) -> Tensor:
        tgt_padding_mask = self.make_padding_mask(tgt_tokens)
        src_padding_mask = self.make_padding_mask(src_tokens)
        tgt_mask = self.make_causal_mask(tgt_tokens.size(1), tgt_tokens.device)

        x = self.tgt_embedding(tgt_tokens)
        x = self.dropout(self.positional_encoding(x))
        return self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor) -> Tensor:
        memory = self.encode(src_tokens)
        decoder_output = self.decode(tgt_tokens, memory, src_tokens)
        return self.lm_head(decoder_output)

    @torch.no_grad()
    def greedy_decode(
        self,
        src_tokens: Tensor,
        bos_token_id: int,
        eos_token_id: Optional[int] = None,
        max_new_tokens: int = 50,
    ) -> Tensor:
        self.eval()
        memory = self.encode(src_tokens)

        batch_size = src_tokens.size(0)
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=src_tokens.device,
        )

        for _ in range(max_new_tokens):
            decoder_output = self.decode(generated, memory, src_tokens)
            next_token_logits = self.lm_head(decoder_output[:, -1])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.eq(eos_token_id).all():
                break

        return generated


if __name__ == "__main__":
    config = TransformerConfig(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
    )

    model = Seq2SeqTransformer(config)
    src = torch.randint(1, config.src_vocab_size, (2, 16))
    tgt = torch.randint(1, config.tgt_vocab_size, (2, 12))
    logits = model(src, tgt)

    print(logits.shape)
