# This provides PLU activation layer, which is not integrated in pytorch yet.
# Reference: https://www.groundai.com/project/plu-the-piecewise-linear-unit-activation-function/1
import torch


def PLU(x, alpha=0.1, c=1.0):
    out = torch.max(alpha * (x + c) - c, torch.min(alpha * (x - c) + c, x))
    return out
