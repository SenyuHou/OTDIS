# OTDIS: Outlier-Trimmed Dual-Interval Smoothing Loss for Sample Selection in Learning with Noisy Labels

## 1. Dependencies

We implement our methods by PyTorch on NVIDIA A800 GPU. The dependencies are as bellow:

- [PyTorch](https://PyTorch.org/), version = 1.7.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.0
- [Anaconda3](https://www.anaconda.com/)

## 2. Preparing conda environments for OTDIS

```bash
conda env create -f environment.yml
conda activate OTDIS-env
conda env update --file environment.yml --prune
```

## 3.Experiments

Default values for input arguments are given in the code. An example command is given:

```bash
python train_on_synthetic_noise.py --dataset cifar10 --model_type CNN --noise_rate 0.3 --noise_type symmetric --gamma 0.5 --gpu 0
```
