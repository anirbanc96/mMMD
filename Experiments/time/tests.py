import numpy as np
from kernels import *


class mMMD_test_statistic:
    """Calculates the test statistic of mMMD."""
    def __init__(self, kernel=None):

        # Initialize kernel here with the provided gamma
        if kernel is None:
            self.kernel = RBF_Kernel() # Default gamma if not provided
        else:
            self.kernel = kernel

    def __call__(self, X, Y):
        """
        Calculates the normalised mMMD statistic for a given set of pairs (X_i, Y_i).
        Z is a list of tuples, where each tuple is (X_i, Y_i).
        """

        n = len(X)

        # Calculate kernel matrices
        K_XX = self.kernel(X, X)
        K_XY = self.kernel(X, Y)
        K_YX = self.kernel(Y, X)
        K_YY = self.kernel(Y, Y)

        # Calculate the H matrix
        H_matrix = K_XX - K_XY - K_YX + K_YY

        # Extract the lower triangle (excluding the diagonal)
        lower_triangle_H = np.tril(H_matrix, k=-1)

        # Calculate the weights (i+1)^-1 for i from 1 to n-1
        # The rows of the lower triangle correspond to i from 1 to n-1
        weights = (np.arange(2, n + 1).astype(float)**(-1)).reshape(-1, 1)

        # Sum the lower triangle along the columns to get Sum_{j=0 to i-1} H_ij for each i
        sum_H_j = np.sum(lower_triangle_H, axis=1)

        # Calculate the weighted sum
        numerator = np.sum(sum_H_j[1:] * weights.flatten())
        denominator = np.sqrt(np.sum((sum_H_j[1:] * weights.flatten())**2))


        return numerator / denominator

class mmd_test_statistic:
    """Calculates the kernel MMD statistic."""
    def __init__(self, kernel=None):

        # Initialize kernel here with the provided gamma
        if kernel is None:
            self.kernel = RBF_Kernel() # Default gamma if not provided
        else:
            self.kernel = kernel

    def __call__(self, X, Y):
        """
        Calculates the MMD statistic for two sets of samples X and Y.
        """
        n = len(X)
        m = len(Y)

        # Calculate kernel matrices
        K_XX = self.kernel(X, X)
        K_XY = self.kernel(X, Y)
        K_YY = self.kernel(Y, Y)

        # Exclude diagonal elements from K_XX and K_YY
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)


        # Calculate MMD statistic
        # MMD^2 = 1/(n(n-1)) * sum(K_XX) + 1/(m(m-1)) * sum(K_YY) - 2/(nm) * sum(K_XY)
        mmd_sq = (np.sum(K_XX) / (n*(n-1)) + np.sum(K_YY) / (m*(m-1)) - 2 * np.sum(K_XY) / (n * m))

        # Return the square root of the MMD squared, ensuring non-negativity
        return mmd_sq


class CrossMMD:

      def __init__(self, kernel_function=None):
        # Initialize kernel here with the provided gamma
        if kernel_function is None:
            self.kernel_function = RBF_Kernel() # Default gamma if not provided
        else:
            self.kernel_function = kernel_function

      def __call__(self, X, Y):

        n, d = X.shape
        m, d_ = Y.shape
        # sanity check
        assert (d_==d) and (n>=2) and (m>=2)

        n1, m1 = n//2, m//2
        n1_, m1_ = n-n1, m-m1

        X1, X2 = X[:n1], X[n1:]
        Y1, Y2 = Y[:m1], Y[m1:]

        Kxx = self.kernel_function(X1, X2)
        Kyy = self.kernel_function(Y1, Y2)

        Kxy = self.kernel_function(X1, Y2)
        Kyx = self.kernel_function(Y1, X2)

        # compute the numerator
        Ux = Kxx.mean() - Kxy.mean()
        Uy = Kyx.mean() - Kyy.mean()
        U = Ux - Uy
        # compute the denominator
        term1 = (Kxx.mean(axis=1) - Kxy.mean(axis=1) - Ux)**2
        sigX2 = term1.mean()
        term2 = (Kyx.mean(axis=1) - Kyy.mean(axis=1) - Uy)**2
        sigY2 = term2.mean()
        sig = np.sqrt(sigX2/n1 + sigY2/m1)
        if not sig>0:
          print(f'term1={term1}, term2={term2}, sigX2={sigX2}, sigY2={sigY2}')
          raise Exception(f'The denominator is {sig}')
        # obtain the statistic
        T = U/sig
        return T

class LMMD:
    """Calculates the test statistic T_n,alpha."""
    def __init__(self, kernel=None):
        # Initialize kernel here with the provided gamma
        if kernel is None:
            self.kernel = RBF_Kernel() # Default gamma if not provided
        else:
            self.kernel = kernel

    def __call__(self, X, Y):
        """
        Calculates the self-normalized statistic from the vector H[2i-1, 2i].
        """

        n = len(X)

        # Calculate kernel matrices
        K_XX = self.kernel(X, X)
        K_XY = self.kernel(X, Y)
        K_YX = self.kernel(Y, X)
        K_YY = self.kernel(Y, Y)

        # Calculate the H matrix
        H_matrix = K_XX - K_XY - K_YX + K_YY

        # Extract the elements H[2i-1, 2i] for i from 1 to n/2
        # The range should be up to n // 2 (exclusive of n // 2 + 1)
        h_values = np.array([H_matrix[2*i, 2*i + 1] for i in range(n // 2)])

        # Calculate the sum and standard deviation
        mean_h = np.mean(h_values)
        std_h = np.sqrt(np.mean(h_values**2))

        return np.sqrt(len(h_values))*mean_h/std_h

class BTest:
    def __init__(self, kernel_function = None, B=None): # Modified to require a kernel

        if kernel_function is None:
            self.kernel_function =RBF_kernel()
        else:
            self.kernel_function = kernel_function

        self.B = B

    def _h_function(self, K_xx, K_yy, K_xy, K_yx):
        return K_xx + K_yy - K_xy - K_yx

    def __call__(self, X, Y):
        n, d = X.shape
        if n != len(Y):
            raise ValueError("Samples X and Y must have the same number of observations.")

        B = self.B if self.B is not None else int(np.sqrt(n))
        if B < 2 or B > n:
            raise ValueError("Block size B must be between 2 and the number of samples.")

        num_blocks = n // B

        K_xx = self.kernel_function(X, X)
        K_yy = self.kernel_function(Y, Y)
        K_xy = self.kernel_function(X, Y)
        h_matrix = self._h_function(K_xx, K_yy, K_xy, K_xy.T)

        block_mmds = np.zeros(num_blocks)

        for i in range(num_blocks):
            start = i * B
            end = start + B
            h_block = h_matrix[start:end, start:end]
            sum_off_diagonal = np.sum(h_block) - np.trace(h_block)
            block_mmds[i] = sum_off_diagonal / (B * (B - 1))

        eta_k_hat = np.mean(block_mmds)
        var_eta_k_hat = np.var(block_mmds, ddof=1) / num_blocks

        if var_eta_k_hat <= 0:
            return 0.0

        test_statistic = eta_k_hat / np.sqrt(var_eta_k_hat)
        return test_statistic