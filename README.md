# Conditional Motion In-Betweening (CMIB)

Official implementation of paper: Conditional Motion In-betweeening.

[Paper](https://www.sciencedirect.com/science/article/pii/S0031320322003752) | [Project Page](https://jihoonerd.github.io/Conditional-Motion-In-Betweening/) | [YouTube](https://youtu.be/XAELcHOREJ8)

<p align="center">
  <img src="assets/graphical_abstract.jpg" alt="Graphical Abstract"/>
</p>

<table>
  <tr>
    <th>in-betweening</th>
    <th>pose-conditioned</th>
  </tr>
  <tr>
    <td><img src="assets/ib.gif"/></td>
    <td><img src="assets/pc.gif"/></td>
  </tr>
</table>

<table>
  <tr>
    <th>walk</th>
    <th>jump</th>
    <th>dance</th>
  </tr>
  <tr>
    <td><img src="assets/walk.gif"/></td>
    <td><img src="assets/jump.gif"/></td>
    <td><img src="assets/dance.gif"/></td>
  </tr>
</table>

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

3. Install appropriate `pytorch` version depending on your device(CPU/GPU), then install packages listed in `requirements.txt`. .

## Trained Weights

You can download trained weights from [here](https://drive.google.com/drive/folders/1_cAhuBxbic3rgPdyrR49kvMnA263bYmi?usp=sharing).

## Train from Scratch

Trining script is `trainer.py`.

```bash
python trainer.py \
	--processed_data_dir="processed_data_80/" \
	--window=90 \
	--batch_size=32 \
	--epochs=5000 \
	--device=0 \
	--entity=cmib_exp \
	--exp_name="cmib_80" \
	--save_interval=50 \
	--learning_rate=0.0001 \
	--loss_cond_weight=1.5 \
	--loss_pos_weight=0.05 \
	--loss_rot_weight=2.0 \
	--from_idx=9 \
	--target_idx=88 \
	--interpolation='slerp'

```

## Inference

You can use `run_cmib.py` for inference. Please refer to help page of `run_cmib.py` for more details.

```python
python run_cmib.py --help
```

## Reference

* LAFAN1 Dataset
  ```
  @article{harvey2020robust,
  author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title     = {Robust Motion In-Betweening},
  booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  publisher = {ACM}, 
  volume    = {39},
  number    = {4},
  year      = {2020}
  }
  ```

## Citation
```
@article{KIM2022108894,
title = {Conditional Motion In-betweening},
journal = {Pattern Recognition},
pages = {108894},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108894},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003752},
author = {Jihoon Kim and Taehyun Byun and Seungyoun Shin and Jungdam Won and Sungjoon Choi},
keywords = {motion in-betweening, conditional motion generation, generative model, motion data augmentation},
abstract = {Motion in-betweening (MIB) is a process of generating intermediate skeletal movement between the given start and target poses while preserving the naturalness of the motion, such as periodic footstep motion while walking. Although state-of-the-art MIB methods are capable of producing plausible motions given sparse key-poses, they often lack the controllability to generate motions satisfying the semantic contexts required in practical applications. We focus on the method that can handle pose or semantic conditioned MIB tasks using a unified model. We also present a motion augmentation method to improve the quality of pose-conditioned motion generation via defining a distribution over smooth trajectories. Our proposed method outperforms the existing state-of-the-art MIB method in pose prediction errors while providing additional controllability. Our code and results are available on our project web page: https://jihoonerd.github.io/Conditional-Motion-In-Betweening}
}
```

## Author

* [Jihoon Kim](https://github.com/jihoonerd)
* [Taehyun Byun](https://github.com/childtoy)
