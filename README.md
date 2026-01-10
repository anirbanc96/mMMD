# Codes for experiments in [*A Martingale Kernel Two-Sample Test*](https://arxiv.org/abs/2510.11853)

This repository contains codes for experiments conducted in the paper [A Martingale Kernel Two-Sample Test](https://arxiv.org/abs/2510.11853). Reproducible codes for the experiments are given in the `Experiments` folder, which is organised in seven sub-folders each for seperate experiments. 

# Repositorty Structure.

- `Experiments\MNIST` : Experiments evaluating the mMMD test (and baselines) on the MNIST dataset. This directory contains two subfolders:
  - **comparison**: Experiments comparing mMMD with baseline methods, corresponding to Section 6.3.
  - **gamma**: Experiments studying performance across the family of test statistics $T_{n,\gamma}$, as presented in Section B.1.
