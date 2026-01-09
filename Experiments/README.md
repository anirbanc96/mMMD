# Experiments

This folder contains the experimental code, configs, and results used to evaluate the mMMD method. Each subfolder focuses on a specific set of experiments.

## Structure

- MNIST
  - Experiments that run the mMMD (and baselines) on the MNIST dataset. This folder is divided into two-subfolders.

- distribution
  - Synthetic distribution experiments used to validate behavior of the test statistics and estimators under controlled distribution shifts.

- minimax
  - Minimax-style experiments (adversarial/robust optimization) used to evaluate worst-case performance and stability of estimators.

- mmMMD
  - Experiments for the mmMMD variant of the method (multi-kernel / multi-modal setups). Contains experiments comparing mmMMD with baseline approaches.

- power
  - Statistical power experiments that measure the detection ability of the tests across effect sizes and sample sizes.

- time
  - Runtime and scalability experiments (time vs. sample size, time vs. dimensionality). Useful for profiling and measuring practical performance.

- typeI
  - Type I error / false positive rate experiments to verify test calibration under the null hypothesis.


## How to run

General steps to run experiments (each subfolder may include its own README or runnable scripts with more specific options):

1. Create and activate a Python environment (recommended):
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run an experiment from a subfolder. For example (from repository root):
   cd Experiments/MNIST
   python run_experiment.py --config configs/example.yaml

4. Results and logs
   - Each experiment should save outputs (models, logs, figures) to an outputs/ or results/ directory inside the subfolder. Check the corresponding script for exact paths.

Notes:
- Some experiments may download datasets automatically (e.g., MNIST) or expect them in a `data/` directory at the repository root. Check the subfolder scripts for details.
- To ensure reproducibility, set random seeds where provided (look for a `seed` option in configs/ or script arguments).
- For heavy experiments that require GPUs, launch the script with CUDA available and set device arguments where supported.

## Reproducing results

- If you plan to reproduce the figures or tables from the paper, start from the configuration files included in the relevant experiment folder and run the corresponding `run_*.py` or evaluation scripts. Consider using smaller toy configs for debugging before running full-scale experiments.

## Contributing

If you add new experiments, please:
- Put them in a new subfolder under Experiments with a short README explaining the purpose, how to run, and where outputs are stored.
- Add a reference to the new subfolder in this README.

## Contact

For questions about the experiments or reproducing results, open an issue in this repository or contact the maintainer: anirbanc96.
