# Conditional Motion In-Betweening (CMIB)

Official implementation of paper: Conditional Motion In-betweeening.

[Paepr]() | [Project Page](https://jihoonerd.github.io/Conditional-Motion-In-Betweening/) | [YouTube](https://youtu.be/nEzzzBeek_E )

<p align="center">
  <img src="assets/graphical_abstract.jpg" alt="Graphical Abstract"/>
</p>

## Environments

This repo is tested on following environment:

* Ubuntu 20.04
* Python >= 3.7
* PyTorch == 1.10.1
* Cuda V11.3.109

## Install

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

You can download trained weights at [here](https://drive.google.com/drive/folders/1_cAhuBxbic3rgPdyrR49kvMnA263bYmi?usp=sharing).


## Train from Scratch


## Inference



## Author

* [Jihoon Kim](https://github.com/jihoonerd)
* [Taehyun Byun](https://github.com/childtoy)