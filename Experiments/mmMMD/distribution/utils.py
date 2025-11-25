from scipy.stats import norm
from tqdm import tqdm
import numpy as np
from numpy.random import SeedSequence, default_rng
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import chi2
from tests import *
from kernels import *

# functions for empirical distribution of mMMD statistic

def compute_single_mMMD_stat(seed, X_distribution, Y_distribution, n_sample_size, gamma_value, kernel = None):

    rng = np.random.default_rng(seed)

    # Sample from distributions
    X_sample = X_distribution.rvs(size=n_sample_size, random_state = rng)
    Y_sample = Y_distribution.rvs(size=n_sample_size, random_state = rng)

    # Ensure 2D
    if X_sample.ndim == 1:
        X_sample = X_sample[:, np.newaxis]
    if Y_sample.ndim == 1:
        Y_sample = Y_sample[:, np.newaxis]

    pooled = np.concatenate((X_sample, Y_sample))
    current_gamma = gamma_value
    if current_gamma is None:
        dists = np.sqrt(np.sum((pooled[:, None, :] - pooled[None, :, :])**2, axis=2))
        median_dist = np.median(dists[dists > 1e-6])
        current_gamma = 1.0 / (2 * median_dist**2) if median_dist > 0 else 1.0

    if kernel is None:
        kernel = RBF_Kernel
    else:
        kernel = kernel

    mult = 2.0 ** np.array([0, 1, 2])
    kernels = [kernel(gamma=m * current_gamma) for m in mult]
    mMMD_stat = multiple_mMMD_test_statistic(kernels=kernels)
    stat_val = mMMD_stat(X_sample, Y_sample)
    return stat_val


def empirical_mMMD_distribution(
    X_distribution,
    Y_distribution,
    n_sample_size=100,
    num_runs=100,
    gamma_value=None,
    n_jobs=-1,
    random_state=None,
    kernel = None
):
    """
    Compute the empirical distribution of the mMMD test statistic in parallel.

    Args:
        X_distribution, Y_distribution: Distributions to sample from.
        n_sample_size (int): Number of samples per distribution.
        num_runs (int): Number of mMMD computations.
        gamma_value: Kernel bandwidth parameter.
        n_jobs (int): Parallel jobs.
        random_state (int or np.random.Generator or None): Seed or generator for reproducibility.

    Returns:
        np.ndarray: Array of mMMD statistics of length num_runs.
    """
    # Create a base SeedSequence from the random_state
    if isinstance(random_state, (int, np.integer)):
        seed_seq = np.random.SeedSequence(random_state)
    elif isinstance(random_state, np.random.Generator):
        seed_seq = np.random.SeedSequence(random_state.integers(1 << 32))
    elif random_state is None:
        seed_seq = np.random.SeedSequence()
    else:
        raise ValueError("random_state must be an int, np.random.Generator, or None")

    # Spawn independent child seeds for each run
    seeds = seed_seq.spawn(num_runs)

    # Run in parallel using the individual seeds
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_mMMD_stat)(
            seed, X_distribution, Y_distribution, n_sample_size, gamma_value, kernel
        )
        for seed in seeds
    )

    return np.array(results)


def plot_empirical_distribution(statistics, figure_size, bins=30, title="Empirical distribution of mMMD statistic", save_path = None):
    """
    Plot histogram of empirical mMMD statistics with standard Gaussian overlaid.

    Args:
        statistics (np.ndarray): Empirical mMMD statistics.
        bins (int): Number of bins for histogram.
        title (str): Plot title.
    """
    plt.figure(figsize = figure_size)

    # Plot histogram and KDE
    sns.histplot(statistics, bins=bins, color='skyblue', stat='density', label='Empirical')

    # Plot standard Gaussian (mean=0, std=1)
    x = np.linspace(min(statistics), max(statistics), 500)
    gaussian_pdf = norm.pdf(x, loc=0, scale=1)
    plt.plot(x, gaussian_pdf, color='red', linestyle='--', label='N(0,1)')

    plt.xlabel("mMMD Statistic")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=1200)
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_overlayed_distributions(
    statistics1,
    statistics2,
    df=1,
    figure_size=(8, 5),
    bins=30,
    label1="Empirical 1",
    label2="Empirical 2",
    title="Empirical Distributions of mMMD statistics",
    show_chi2=True,
    save_path=None
):
    """
    Plot two empirical distributions overlaid with optional chi-squared distribution.

    Args:
        statistics1 (np.ndarray): First set of statistics.
        statistics2 (np.ndarray): Second set of statistics.
        df (int): Degrees of freedom for chi-squared distribution.
        figure_size (tuple): Figure size for the plot.
        bins (int): Number of bins for histograms.
        label1 (str): Label for first histogram.
        label2 (str): Label for second histogram.
        title (str): Title of the plot.
        show_chi2 (bool): Whether to overlay chi-squared distribution.
        save_path (str or None): Path to save the figure, if provided.
    """
    plt.figure(figsize=figure_size)

    # Plot both histograms
    sns.histplot(statistics1, bins=bins, stat='density', color='skyblue', label=label1, alpha=0.8)
    sns.histplot(statistics2, bins=bins, stat='density', color='orange', label=label2, alpha=0.4)

    # Optionally add chi-squared distribution
    if show_chi2:
        x_min = min(np.min(statistics1), np.min(statistics2))
        x_max = max(np.max(statistics1), np.max(statistics2))
        x = np.linspace(x_min, x_max, 500)
        chi2_pdf = chi2.pdf(x, df=df)
        plt.plot(x, chi2_pdf, color='red', linestyle='--', label=f'$\chi^2_{{{df}}}$')

    # Plot formatting
    plt.xlabel("Statistic Value")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=1200)
        print(f"Figure saved to: {save_path}")

    plt.show()