import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dimension=256, max_len=50, device="cpu"):
        super().__init__()
        self.device = device
        pe = torch.zeros(max_len, dimension, device=self.device)
        position = torch.arange(max_len, 0, -1, device=self.device).unsqueeze(
            1
        )  # It's a time to arrival(TTA), so decrease it from max_len.
        div_term = torch.exp(
            torch.arange(0, dimension, 2, device=self.device).float()
            * (-math.log(10000.0) / dimension)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # 3.3 The model sees a constant ztta for 5 frames before it starts to vary.
        ztta_const_part = self.pe[0][5]
        self.pe[0][0] = ztta_const_part
        self.pe[0][1] = ztta_const_part
        self.pe[0][2] = ztta_const_part
        self.pe[0][3] = ztta_const_part
        self.pe[0][4] = ztta_const_part

    def forward(self, x, time_step: int):
        x = x + self.pe[:, time_step]
        return x
