import numpy as np

class RBF_Kernel:
    """Implements a Radial Basis Function (RBF) kernel."""
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y):
        """Calculates the kernel value between two vectors or sets of vectors."""
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim == 1 and y.ndim == 1:
            return np.exp(-self.gamma * np.linalg.norm(x - y)**2)
        elif x.ndim == 2 and y.ndim == 2:
            # Calculate pairwise squared Euclidean distances using broadcasting
            # (x_i - y_j)^2 = x_i^2 - 2*x_i*y_j + y_j^2
            x_sq = np.sum(x**2, axis=1, keepdims=True)
            y_sq = np.sum(y**2, axis=1, keepdims=True).T
            pairwise_sq_dist = x_sq - 2 * x @ y.T + y_sq
            return np.exp(-self.gamma * pairwise_sq_dist)
        else:
            raise ValueError("Inputs to kernel must be 1D or 2D arrays.")


class Laplace_Kernel:
    """Implements a Laplace (exponential) kernel."""
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y):
        """Calculates the kernel value between two vectors or sets of vectors."""
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim == 1 and y.ndim == 1:
            return np.exp(-self.gamma * np.linalg.norm(x - y, ord=2))

        elif x.ndim == 2 and y.ndim == 2:
            # Compute pairwise Euclidean distances using broadcasting
            # Efficient computation of ||x_i - y_j|| for all pairs
            x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
            y_norm = np.sum(y**2, axis=1).reshape(1, -1)
            sq_dist = x_norm - 2 * x @ y.T + y_norm
            dist = np.sqrt(np.maximum(sq_dist, 0))  # ensure non-negative before sqrt
            return np.exp(-self.gamma * dist)

        else:
            raise ValueError("Inputs to kernel must be 1D or 2D arrays.")
