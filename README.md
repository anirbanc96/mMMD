# A Martingale Kernel Two-Sample Test

[![Paper](https://img.shields.io/badge/arXiv-2510.11853-b31b1b.svg)](https://arxiv.org/abs/2510.11853)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the experimental code for the paper **[*A Martingale Kernel Two-Sample Test*](https://arxiv.org/abs/2510.11853)**.

All experiments reported in the paper are fully reproducible. The code is organized by experiment type under the `Experiments` directory.

---

## Repository Structure

```text
├── Experiments/
│   ├── MNIST/
│   │   ├── comparison/
│   │   └── gamma/
│   ├── distribution/
│   ├── minimax/
│   ├── mmMMD/
│   │   ├── distribution/
│   │   └── power/
│   ├── power/
│   ├── time/
│   └── typeI/
```


### Description of Folders

- **`Experiments/MNIST`** : Experiments evaluating the mMMD test, baseline methods, and the general family  
  $T_{n,\gamma}$ on the MNIST dataset.
  - `comparison`: Performance comparison of mMMD against baseline tests (Section 6.3).
  - `gamma`: Effect of varying $\gamma$ in the family $T_{n,\gamma}$ (Section B.1).
- **`Experiments/distribution`** : Validation of the null distribution of the mMMD statistic (Section 6.1).
- **`Experiments/minimax`** : Performance comparison of mMMD and baseline methods in the minimax regime (Section B.3).
- **`Experiments/mmMMD`** : Experiments for the multi-kernel variant (mmMMD) introduced in Section 7.1.
  - `distribution`: Validation of the null distribution of the mmMMD statistic.
  - `power`: Empirical power comparison with baseline methods (Figure 7).
- **`Experiments/power`** : Power comparison between the mMMD test and baseline methods (Section 6.2).
- **`Experiments/time`** : Runtime comparisons between mMMD and baseline methods (Figure 1(c)).
- **`Experiments/typeI`** : Type-I error control experiments for mMMD and baseline methods (Section B.2).

---

## Requirements

- Python 3.8+
- Jupyter Notebook

Additional dependencies are specified within individual notebooks.

---

## Running the Experiments

Each experiment folder contains one or more Jupyter notebooks (`.ipynb` files).

To reproduce the results:
1. Navigate to the desired experiment folder.
2. Open the corresponding notebook.
3. Run all cells to generate the figures and results reported in the paper.

---
