# Codes for experiments in [*A Martingale Kernel Two-Sample Test*](https://arxiv.org/abs/2510.11853)

This repository contains codes for experiments conducted in the paper [A Martingale Kernel Two-Sample Test](https://arxiv.org/abs/2510.11853). Reproducible codes for the experiments are given in the `Experiments` folder, which is organised in seven sub-folders each for seperate experiments. 

## Repositorty Structure.

- `Experiments/MNIST` : Experiments evaluating the mMMD, baseline tests and the general family $T_{n,\gamma}$ on the MNIST dataset. This directory further contains two sub-folders, `comparison` evaluating performance of mMMD test against baselines using MNIST dataset (see Section 6.3) and `gamma` where we compare performance across $\gamma$ for the general family $T_{n,\gamma}$ on the MNIST dataset (see Section B.1).
- `Experiments/distribution` : Experiments validating the null distribution of the mMMD statistic (see Section 6.1).
- `Experiments/minimax` : Experiments comparing the performance of the mMMD test with baseline methods in the minimax regime (see Section B.3).
- `Experiments/mmMMD` : Experiments for the multi-kernel variant (mmMMD) defined in Section 7.1. This folder is further sub-dvided into two folders, `distribution` validating the null distribution of mmMMD statistic and `power` comparing empirical power of mmMMD test against baselines (see Figure 7).
- `Experiments/power` : Experiments comparing the power of the mMMD test against baseline methods (see Section 6.2).
- `Experiments/time` : Runtime comparison experiments between mMMD and baseline methods (see Figure 1(c)).
- `Experiments/typeI` : Experiments evaluating type-I error control of the mMMD test and baseline methods(see Section B.2).

## Running the experiments

All experiment folders contains a `.ipynb` running which produces the corresponding images. 
