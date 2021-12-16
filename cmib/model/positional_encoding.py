import torch
from torch import nn, Tensor
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int = 32, d_model: int = 96):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)

    def forward(self, inputs):
        positions = (
            torch.arange(inputs.size(0), device=inputs.device)
            .expand(inputs.size(1), inputs.size(0))
            .contiguous()
            + 1
        )
        outputs = inputs + self.pos_emb(positions).permute(1, 0, 2)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
