import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dimension=256, max_len=30, device="cpu"):
        super().__init__()
        self.device = device
        pe = torch.zeros(max_len, dimension, device=self.device)
        position = torch.arange(0, max_len, step=1, device=self.device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, dimension, 2, device=self.device).float()
            * (-math.log(10000.0) / dimension)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # 3.3 This means that when dealing with transitions of length T max (trans), the model sees a constant
        # z tta for 5 frames before it starts to vary.
        ztta_const_part = self.pe[0][max_len - 5]
        self.pe[0][max_len - 4] = ztta_const_part
        self.pe[0][max_len - 3] = ztta_const_part
        self.pe[0][max_len - 2] = ztta_const_part
        self.pe[0][max_len - 1] = ztta_const_part

    def forward(self, x, tta: int):
        x = x + self.pe[:, tta - 1]
        return x