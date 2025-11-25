from scipy.stats import norm
from tqdm import tqdm
import numpy as np
from numpy.random import SeedSequence, default_rng
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import time
from tests import *

def run_single_test_timing(X_distribution, Y_distribution, n_sample_size, Kernel = None, gamma_value=None):

    # Sample data
    X_sample = X_distribution.rvs(size=n_sample_size)
    Y_sample = Y_distribution.rvs(size=n_sample_size)

    # Ensure 2D
    if X_sample.ndim == 1:
        X_sample = X_sample[:, np.newaxis]
    if Y_sample.ndim == 1:
        Y_sample = Y_sample[:, np.newaxis]

    pooled_XY = np.concatenate((X_sample, Y_sample))

    # Compute gamma
    current_gamma = gamma_value
    if current_gamma is None:
        z_sq = np.sum(pooled_XY**2, axis=1, keepdims=True)
        pairwise_sq_dist = z_sq - 2 * pooled_XY @ pooled_XY.T + z_sq.T
        pairwise_sq_dist[pairwise_sq_dist < 0] = 0
        pairwise_distances_Z = np.sqrt(pairwise_sq_dist).flatten()
        non_zero_distances_Z = pairwise_distances_Z[pairwise_distances_Z > 1e-6]
        if len(non_zero_distances_Z) > 0:
            median_bandwidth_Z = np.median(non_zero_distances_Z)
            current_gamma = 1.0 / (2 * median_bandwidth_Z**2)
        else:
            current_gamma = 1.0

    if Kernel is None:
        Kernel = RBF_Kernel
    else:
        Kernel = Kernel

    kernel = Kernel(gamma=current_gamma)

    runtimes = {}

    # --- mMMD ---
    start = time.perf_counter()
    _ = mMMD_test_statistic(kernel=kernel)(X_sample, Y_sample)
    p_value_mMMD = 1.0 - norm.cdf(_)  # simulate full test
    runtimes["mMMD"] = time.perf_counter() - start

    # --- mmd (includes permutation test) ---
    start = time.perf_counter()
    mmd_stat = mmd_test_statistic(kernel=kernel)
    mmd_observed = mmd_stat(X_sample, Y_sample)
    pooled = np.concatenate((X_sample, Y_sample))
    n = len(X_sample)
    mmd_permutations = []
    for _ in range(200):
        permuted = np.random.permutation(pooled)
        X_perm, Y_perm = permuted[:n], permuted[n:]
        mmd_permutations.append(mmd_stat(X_perm, Y_perm))
    p_value_mmd = (np.sum(np.array(mmd_permutations) >= mmd_observed) + 1) / (201)
    runtimes["mmd"] = time.perf_counter() - start

    # --- LMMD ---
    start = time.perf_counter()
    _ = LMMD(kernel=kernel)(X_sample, Y_sample)
    p_value_LMMD = 1.0 - norm.cdf(_)  # simulate full test
    runtimes["LMMD"] = time.perf_counter() - start

    # --- BTest ---
    start = time.perf_counter()
    _ = BTest(kernel_function=kernel)(X_sample, Y_sample)
    p_value_BTest = 1.0 - norm.cdf(_)  # simulate full test
    runtimes["BTest"] = time.perf_counter() - start

    # --- CrossMMD ---
    start = time.perf_counter()
    _ = CrossMMD(kernel_function=kernel)(X_sample, Y_sample)
    p_value_CrossMMD = 1.0 - norm.cdf(_)  # simulate full test
    runtimes["CrossMMD"] = time.perf_counter() - start

    return runtimes


def compute_test_runtimes(X_distribution, Y_distribution, sample_sizes,
                          num_runs=100, gamma_value=None, Kernel = None, n_jobs=-1):
    
    test_names = ['mMMD', 'mmd', 'LMMD', 'BTest', 'CrossMMD']
    timing_results = {test: {'mean': [], 'std': []} for test in test_names}

    for n_sample_size in sample_sizes:
        run_times = Parallel(n_jobs=n_jobs)(
            delayed(run_single_test_timing)(
                X_distribution, Y_distribution, n_sample_size, gamma_value
            ) for _ in tqdm(range(num_runs), desc=f"Timing sample size {n_sample_size}")
        )

        # Convert list of dicts into dict of lists
        times_by_test = {test: [] for test in test_names}
        for result in run_times:
            for test in test_names:
                times_by_test[test].append(result[test])

        # Compute mean/std for each test
        for test in test_names:
            timing_results[test]['mean'].append(np.mean(times_by_test[test]))
            timing_results[test]['std'].append(np.std(times_by_test[test]))

    return timing_results

def plot_test_runtimes(runtime_results, sample_sizes):

    plt.figure(figsize=(10, 6))
    for test_name, results in runtime_results.items():
        means = np.array(results['mean'])
        stds = np.array(results['std'])
        plt.plot(sample_sizes, means, label=test_name)
        plt.fill_between(sample_sizes, means - stds, means + stds, alpha=0.2)

    plt.xlabel("Sample Size")
    plt.ylabel("Runtime per Test (seconds)")
    plt.title("Runtime vs Sample Size (100 runs per point)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_runtime_table(runtime_results, sample_sizes, save_path=None):
    # Prepare table headers
    test_names = list(runtime_results.keys())
    headers = ["Test / Sample Size"] + [str(s) for s in sample_sizes]

    # Prepare table data: mean ± std for each test and sample size
    table_data = []
    for test in test_names:
        means = np.array(runtime_results[test]['mean'])
        stds = np.array(runtime_results[test]['std'])
        row = [test] + [f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)]
        table_data.append(row)

    # Calculate compact figure size dynamically
    cell_width = 1.2
    cell_height = 0.45
    fig_width = cell_width * (len(sample_sizes) + 1)
    fig_height = cell_height * (len(test_names)) + 0.25

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.axis("tight")

    # Create table
    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="upper center",
        cellLoc="center"
    )

    # Font and layout
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)  # Smaller font for tighter layout
    tbl.scale(1.0, 1.0)   # No upscaling
    plt.tight_layout(pad=0)
    plt.title("Runtime vs Sample Size\n(Mean ± Std in seconds)", fontsize=11)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=1200)
        print(f"Figure saved to: {save_path}")

    plt.show()


