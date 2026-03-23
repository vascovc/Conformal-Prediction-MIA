# project/mica/distance_matrix.py
import numpy as np
from typing import Tuple

def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances (n x n).
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    # Efficient computation: (x - y)^2 = x^2 + y^2 - 2xy
    sum_sq = np.sum(np.square(X), axis=1)
    D2 = sum_sq[:, None] + sum_sq[None, :] - 2 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def distances_to_point(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from each row in X to x (shape (n,))
    """
    diff = X - x.reshape(1, -1)
    return np.sqrt(np.sum(diff**2, axis=1))
