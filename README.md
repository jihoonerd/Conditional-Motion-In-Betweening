# Doubliy Conditioned Motion In-Painting

WIP Project

## Setup

1. Follow [`LAFAN1`](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) dataset's installation guide.
   *You need to install git lfs first before cloning the dataset repo.*

2. Run LAFAN1's `evaluate.py` to unzip and validate it. (Install `numpy` first if you don't have it)
   ```bash
   $ pip install numpy
   $ python ubisoft-laforge-animation-dataset/evaluate.py 
   ```
   With this, you will have unpacked LAFAN dataset under `ubisoft-laforge-animation-dataset` folder.

3. Now, install packages listed in `requirements.txt`. Use appropriate `pytorch` version depending on your device(CPU/GPU).

## Trained Weights

| 30 Frame Model Weight | 40 Frame Model Weight |
|:---:|:---:|
|[Download](https://drive.google.com/file/d/1BuyHzDrgbvFIDlmfBnKo91UjmGQELA0w/view?usp=sharing)|[Download](https://drive.google.com/file/d/1C2u-Hd3w7PD_hp3REAZrMXI5A4NBkW-3/view?usp=sharing)|

