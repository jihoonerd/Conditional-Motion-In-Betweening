import numpy as np
import torch
from rmi.model.noise_injector import noise_injector
from rmi.model.plu import PLU
from rmi.model.positional_encoding import PositionalEncoding


def test_plu_activation():
    input_sample = torch.Tensor(np.linspace(-2, 2, num=41))
    out = PLU(input_sample)
    assert out[0] == 0.1 * (-2 + 1) - 1
    assert out[-1] == 0.1 * (2 - 1) + 1

def test_noise_injector():
    input_sample = np.arange(0, 50)
    noise = [noise_injector(i, 50) for i in input_sample]
    assert np.all(noise[:20] == np.array([1] * 20))
    assert (noise[20] - noise[-5]) == 1.0
    assert np.all(noise[-5:] == np.array([0] * 5))

def test_tta():
    input_sample = torch.randn(128, 256)
    pe = PositionalEncoding(dimension=256, max_len=50)
    out = pe(input_sample, 3)
    assert np.all(out.numpy() == (pe.pe[0][3] + input_sample).numpy())
