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

# functions for evaluating empirical power

def run_tests(X, Y, gamma, num_permutations=200, Kernel = None, multiplier_list = None):
    """
    Runs all implemented tests and returns observed statistics and p-values.

    Args:
        X (np.ndarray): First sample.
        Y (np.ndarray): Second sample.
        alpha (float): The alpha value for the TestStatistic.
        gamma (float): The gamma value for the kernel.
        num_permutations (int): Number of permutations for the MMD test.

    Returns:
        tuple: Observed statistics and p-values for mMMD, mmd, LMMD, and BTest.
    """
    if Kernel is None:
        Kernel = RBF_Kernel
    else:
        Kernel = Kernel

    if multiplier_list is None:
        multiplier_list = 2.0**np.array([0, 1, 2])
    else:
        multiplier_list = multiplier_list

    kernel = Kernel(gamma=gamma)

    # mMMD Test
    mMMD_statistic = mMMD_test_statistic(kernel=kernel)
    mMMD_observed = mMMD_statistic(X, Y)
    p_value_mMMD = 1.0 - norm.cdf(mMMD_observed) # One-sided Gaussian test

    # multiple mMMD test with all gaussian kernels
    multiple_kernels_g = [RBF_Kernel(gamma = multiplier * gamma) for multiplier in multiplier_list]
    multiple_mMMD_statistic_g = multiple_mMMD_test_statistic(kernels = multiple_kernels_g)
    multiple_mMMD_observed_g = multiple_mMMD_statistic_g(X, Y)
    p_value_multiple_mMMD_g = 1.0-chi2.cdf(multiple_mMMD_observed_g, df=len(multiple_kernels_g))

    # multiple mMMD test with mixture of gaussian and laplace kernels
    multiple_kernels_gl = [RBF_Kernel(gamma = gamma), Laplace_Kernel(gamma = gamma)]
    multiple_mMMD_statistic_gl = multiple_mMMD_test_statistic(kernels = multiple_kernels_gl)
    multiple_mMMD_observed_gl = multiple_mMMD_statistic_gl(X, Y)
    p_value_multiple_mMMD_gl = 1.0-chi2.cdf(multiple_mMMD_observed_gl, df=len(multiple_kernels_gl))

    # multiple mMMD test with all gaussian kernels
    multiple_kernels_l = [Laplace_Kernel(gamma = multiplier * gamma) for multiplier in multiplier_list]
    multiple_mMMD_statistic_l = multiple_mMMD_test_statistic(kernels = multiple_kernels_l)
    multiple_mMMD_observed_l = multiple_mMMD_statistic_l(X, Y)
    p_value_multiple_mMMD_l = 1.0-chi2.cdf(multiple_mMMD_observed_l, df=len(multiple_kernels_l))

    # MMD Test (using permutation test)
    # mmd_statistic = mmd_test_statistic(kernel=kernel)
    # mmd_observed = mmd_statistic(X, Y)

    # pooled_XY = np.concatenate((X, Y))
    # n = len(X)
    # mmd_permutations = []
    # for _ in range(num_permutations):
    #     permuted_XY = np.random.permutation(pooled_XY)
    #     X_permuted = permuted_XY[:n]
    #     Y_permuted = permuted_XY[n:]
    #     mmd_permutations.append(mmd_statistic(X_permuted, Y_permuted))
    # p_value_mmd = (np.sum(np.array(mmd_permutations) >= mmd_observed) + 1) / (num_permutations + 1)


    # LMMD Test
    # LMMD_statistic = LMMD(kernel=kernel)
    # LMMD_observed = LMMD_statistic(X, Y)
    # p_value_LMMD = 1.0 - norm.cdf(LMMD_observed) # One-sided Gaussian test


    # BTest
    # BTest_statistic = BTest(kernel_function=kernel) # Pass the kernel function
    # BTest_observed = BTest_statistic(X, Y)
    # p_value_BTest = 1.0 - norm.cdf(BTest_observed) # One-sided Gaussian test

    # CrossMMD
    cross_mmd_statistic = CrossMMD(kernel_function=kernel)
    cross_mmd_observed = cross_mmd_statistic(X, Y)
    p_value_cross_mmd = 1.0 - norm.cdf(cross_mmd_observed) # One-sided Gaussian test


    return p_value_mMMD, p_value_multiple_mMMD_g, p_value_multiple_mMMD_l, p_value_multiple_mMMD_gl, p_value_cross_mmd


def run_empirical_power_experiment(X_distribution, Y_distribution,
                                   significance_level = 0.05, n_sample_size=100, num_runs=100, Kernel = None, gamma_value=None, seed = None):
    """
    Runs the experiment multiple times to calculate empirical power for all tests.

    Args:
        alpha_value (float): The alpha value for the mMMD and LMMD statistics.
        X_distribution: The first distribution to sample from.
        Y_distribution: The second distribution to sample from.
        n_sample_size (int): The size of the sample to draw from each group.
        num_runs (int): The number of times to run the experiment.
        significance_level (float): The significance level for the tests.
        gamma_value (float, optional): The gamma value for the kernel. If None, the median heuristic is used.


    Returns:
        tuple: Empirical power for mMMD, mmd, LMMD, BTest, and CrossMMD.
    """
    p_values_mMMD = []
    p_values_multiple_mMMD_g = []
    p_values_multiple_mMMD_l = []
    p_values_multiple_mMMD_gl = []
    # p_values_mmd = []
    # p_values_LMMD = []
    # p_values_BTest = []
    p_values_cross_mmd = []

    seed_seq = SeedSequence(seed)
    child_seeds = seed_seq.spawn(num_runs)

    for run in range(num_runs):

        rng = default_rng(child_seeds[run])
        
        # Sample n many X from first_digits and n many Y from second_digits
        X_sample = X_distribution.rvs(size=n_sample_size, random_state = rng)
        Y_sample = Y_distribution.rvs(size=n_sample_size, random_state = rng)

        # Ensure samples are at least 2D for consistent processing
        if X_sample.ndim == 1:
            X_sample = X_sample[:, np.newaxis]
        if Y_sample.ndim == 1:
            Y_sample = Y_sample[:, np.newaxis]

        pooled_XY = np.concatenate((X_sample, Y_sample))

        # Compute median heuristic if gamma_value is not provided
        current_gamma = gamma_value
        if current_gamma is None:
          # Calculate pairwise squared Euclidean distances using broadcasting
          # (z_i - z_j)^2 = z_i^2 - 2*z_i*z_j + z_j^2
          z_sq = np.sum(pooled_XY**2, axis=1, keepdims=True)
          pairwise_sq_dist = z_sq - 2 * pooled_XY @ pooled_XY.T + z_sq.T

          # Ensure non-negativity due to potential floating point inaccuracies
          pairwise_sq_dist[pairwise_sq_dist < 0] = 0
          pairwise_distances_Z = np.sqrt(pairwise_sq_dist).flatten()

          non_zero_distances_Z = pairwise_distances_Z[pairwise_distances_Z > 1e-6]
          if len(non_zero_distances_Z) > 0:
              median_bandwidth_Z = np.median(non_zero_distances_Z)
              current_gamma = 1.0 / (2 * median_bandwidth_Z**2) # Common heuristic
          else:
              # Handle case where all distances are zero (e.g., all samples are identical)
              current_gamma = 1.0


        # Run the tests with Gaussian threshold
        p_value_mMMD, p_value_multiple_mMMD_g, p_value_multiple_mMMD_l, p_value_multiple_mMMD_gl, p_value_cross_mmd = run_tests(X_sample, Y_sample, gamma=current_gamma, Kernel = Kernel)
        p_values_mMMD.append(p_value_mMMD)
        p_values_multiple_mMMD_g.append(p_value_multiple_mMMD_g)
        p_values_multiple_mMMD_l.append(p_value_multiple_mMMD_l)
        p_values_multiple_mMMD_gl.append(p_value_multiple_mMMD_gl)
        # p_values_mmd.append(p_value_mmd)
        # p_values_LMMD.append(p_value_LMMD)
        # p_values_BTest.append(p_value_BTest)
        p_values_cross_mmd.append(p_value_cross_mmd)


    # Calculate empirical power for both tests
    empirical_power_mMMD = np.sum(np.array(p_values_mMMD) < significance_level) / num_runs
    empirical_power_multiple_mMMD_g = np.sum(np.array(p_values_multiple_mMMD_g) < significance_level) / num_runs
    empirical_power_multiple_mMMD_l = np.sum(np.array(p_values_multiple_mMMD_l) < significance_level) / num_runs
    empirical_power_multiple_mMMD_gl = np.sum(np.array(p_values_multiple_mMMD_gl) < significance_level) / num_runs
    # empirical_power_mmd = np.sum(np.array(p_values_mmd) < significance_level) / num_runs
    # empirical_power_LMMD = np.sum(np.array(p_values_LMMD) < significance_level) / num_runs
    # empirical_power_BTest = np.sum(np.array(p_values_BTest) < significance_level) / num_runs
    empirical_power_cross_mmd = np.sum(np.array(p_values_cross_mmd) < significance_level) / num_runs



    return empirical_power_mMMD, empirical_power_multiple_mMMD_g, empirical_power_multiple_mMMD_l, empirical_power_multiple_mMMD_gl, empirical_power_cross_mmd


def compute_power_over_sample_sizes(X_distribution, Y_distribution,
                                    sample_sizes, num_outer_runs=100, significance_level=0.05, Kernel = None, gamma_value=None, seed = None, n_jobs=-1):
    """
    Computes empirical power and its standard deviation for different sample sizes using parallel processing.

    Args:
        X_distribution: The first distribution to sample from.
        Y_distribution: The second distribution to sample from.
        sample_sizes (list): A list of sample sizes to test.
        num_runs (int): Number of runs per sample size (used for empirical power estimation).
        significance_level (float): The significance level for hypothesis testing.
        gamma_value (float, optional): Kernel bandwidth. If None, median heuristic is used.
        n_jobs (int): Number of parallel jobs. Default is -1 (use all cores).

    Returns:
        dict: Dictionary containing mean and std deviation of empirical power per test per sample size.
    """

    test_names = ['mMMD', 'multiple_mMMD_g', 'multiple_mMMD_l', 'multiple_mMMD_gl', 'CrossMMD']
    power_results_by_size = {test: {'mean': [], 'std': []} for test in test_names}

    global_seed_seq = SeedSequence(seed)
    rng = np.random.default_rng(global_seed_seq)

    for n_sample_size in sample_sizes:

        child_seeds = rng.integers(low=0, high=2**32 - 1, size=num_outer_runs)

        # Parallel execution of num_runs experiments
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_empirical_power_experiment)(
                X_distribution, Y_distribution,
                significance_level=significance_level,
                n_sample_size=n_sample_size,
                num_runs=100,
                gamma_value=gamma_value,
                Kernel = Kernel,
                seed = child_seeds[_]
            ) for _ in tqdm(range(num_outer_runs), desc=f"Sample size {n_sample_size}")
        )

        # Convert list of tuples into a dict of lists
        results_array = np.array(results)  # Shape: (num_runs, 5)

        for i, test in enumerate(test_names):
            test_results = results_array[:, i]
            power_results_by_size[test]['mean'].append(np.mean(test_results))
            power_results_by_size[test]['std'].append(np.std(test_results))

    return power_results_by_size
