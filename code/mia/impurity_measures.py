# project/mica/impurity_measures.py
import numpy as np
from typing import Iterable

""" Classification impurity measures """

def gini_impurity(labels: Iterable) -> float:
    """
    Compute Gini impurity for a list/array of labels.
    Returns value in [0, 0.5] for binary; generalizes to multi-class in [0, 1 - 1/k].
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

def shannon_entropy(labels: Iterable) -> float:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    # handle zero probs
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def misclassification_error(labels: Iterable) -> float:
    """
    Compute misclassification error for a list/array of labels.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.max(p)

def tsallis_entropy(labels: Iterable, q: float = 2.0) -> float:
    """
    Compute Tsallis entropy for a list/array of labels.
    A parameter q controls the nature of the entropy.
    - As q -> 1, Tsallis entropy approaches Shannon entropy.
    - For q = 2, it is equivalent to Gini impurity.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    
    if np.isclose(q, 1.0):
        # Handle the case where q is close to 1, which should be Shannon entropy
        p = p[p > 0] # Avoid log(0)
        return -np.sum(p * np.log(p))
    else:
        return (1.0 / (q - 1.0)) * (1.0 - np.sum(p**q))

def one_vs_rest_impurity(labels: Iterable):
    """
    Compute impurity for a one-vs-rest scenario.
    Assumes labels are 0 for the class of interest and 1 for all other classes.
    The impurity is simply the proportion of "rest" samples (the mean of the labels).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    # With 0/1 labels, mean is the proportion of 1s.
    return np.mean(labels)

"""---------------------------------------------------------------------------------------------------------------------------------------"""    
""" Regression impurity measures """

def regression_impurity_variance(values: Iterable) -> float:
    """
    Compute impurity for a list/array of continuous values (regression).
    This is simply the variance of the values.
    """
    values = np.asarray(values)
    if values.size < 2:
        return 0.0
    return np.var(values)

def regression_impurity_std(values: Iterable) -> float:
    """
    Compute impurity for a list/array of continuous values (regression).
    This is simply the standard deviation of the values.
    """
    values = np.asarray(values)
    if values.size < 2:
        return 0.0
    return np.std(values)

def mean_absolute_error(values: Iterable) -> float:
    """
    Compute impurity for a list/array of continuous values (regression).
    This is the mean absolute error from the mean of the values.
    """
    values = np.asarray(values)
    if values.size < 2:
        return 0.0
    return np.mean(np.abs(values - np.mean(values)))