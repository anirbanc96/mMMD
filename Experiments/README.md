# Experiments

This directory contains the code and results for the experiments presented in **_A Martingale Kernel Two-Sample Test_**. The experiments are organized into seven subdirectories, each corresponding to a specific experimental setting or evaluation goal discussed in the paper.

## Directory Structure

- **MNIST** : Experiments evaluating the mMMD test (and baselines) on the MNIST dataset. This directory contains two subfolders:
  - **comparison**: Experiments comparing mMMD with baseline methods, corresponding to Section 6.3.
  - **gamma**: Experiments studying performance across the family of test statistics $T_{n,\gamma}$, as presented in Section B.1.

- **distribution** : Experiments validating the null distribution of the mMMD statistic, corresponding to Section 6.1.

- **minimax** : Experiments comparing the performance of the mMMD test with baseline methods in the minimax regime, as presented in Section B.3.

- **mmMMD** : Experiments for the multi-kernel variant (mmMMD), as presented in Section 7.1. This directory contains two subfolders:
  - **distribution**: Validation of the null distribution of the mmMMD statistic.
  - **power**: Power comparisons between mmMMD and baseline methods.

- **power** : Experiments comparing the power of the mMMD test against baseline methods, corresponding to Section 6.2.

- **time** : Runtime comparison experiments between mMMD and baseline methods, corresponding to Figure 1(c).

- **typeI** : Experiments evaluating type-I error control of the mMMD test and baseline methods, corresponding to Section B.2.
