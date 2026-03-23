# project/mica/hypersphere_analysis.py
import numpy as np
from .impurity_measures import gini_impurity, shannon_entropy
from typing import List, Tuple, Callable

def impurities_at_radius(
    X_centers: np.ndarray,
    X_search: np.ndarray,
    y_search: np.ndarray,
    radius: float,
    impurity_fn: Callable = gini_impurity
) -> np.ndarray:
    """
    For each point in X_centers, compute the impurity of labels from y_search
    for neighbors found in X_search within a given radius.

    The impurity function (e.g., Gini for classification, variance for
    regression) is passed as an argument.

    Returns an array of impurities with length equal to the number of points
    in X_centers.
    """
    if not isinstance(X_centers, np.ndarray):
        X_centers = np.array(X_centers)
    if not isinstance(X_search, np.ndarray):
        X_search = np.array(X_search)
    n_centers = X_centers.shape[0]
    impurities = np.zeros(n_centers)
    # naive loop (can be optimized with kd-tree for large n)
    for i in range(n_centers):
        center = X_centers[i]
        dists = np.linalg.norm(X_search - center, axis=1)
        mask = dists <= radius
        if np.any(mask):
            impurities[i] = impurity_fn(y_search[mask])
        else:
            impurities[i] = 0  # or some other default for no neighbors
    return impurities

def radii_sequence(max_radius: float, n_radii: int = 10,raddi_spacing: str = "linear") -> np.ndarray:
    """
    Create an increasing sequence of radii from small positive up to max_radius (inclusive).
    """
    if raddi_spacing == "linear":
        return np.linspace(max_radius / n_radii, max_radius, num=n_radii)
    elif raddi_spacing == "log":
        return np.logspace(np.log10(max_radius / n_radii), np.log10(max_radius), num=n_radii)
