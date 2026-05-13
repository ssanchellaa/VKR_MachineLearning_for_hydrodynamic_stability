# data_utils.py
# Data processing utilities for neural network training


from typing import List, Any, Tuple
import numpy as np
import random

VectorType = list[float]

# ---------- basic statistics
def compute_mean(values: List[Any]) -> float:
    """Calculate arithmetic mean."""
    return sum(values) / len(values)

def compute_stddev(values: List[Any]) -> float:
    """Calculate sample standard deviation."""
    sum_sq = sum(np.power(values, 2))
    mean_sq = np.power(compute_mean(values), 2)
    variance = (sum_sq - len(values) * mean_sq) / (len(values) - 1)
    return np.sqrt(variance)

def print_stats(data: List[Any]):
    """Print mean and standard deviation."""
    print("       mean is", compute_mean(data))
    print("  deviation is", compute_stddev(data))

def compute_covariance(x_vals: List[int], y_vals: List[float]) -> float:
    """Calculate covariance between two lists."""
    assert len(x_vals) == len(y_vals)
    x_mean = compute_mean(x_vals)
    y_mean = compute_mean(y_vals)
    x_dev = [x - x_mean for x in x_vals]
    y_dev = [y - y_mean for y in y_vals]
    cov = sum(x_i * y_i for (x_i, y_i) in zip(x_dev, y_dev)) / (len(x_vals) - 1)
    return cov

def compute_correlation(x_vals: List[int], y_vals: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    std_x = compute_stddev(x_vals)
    std_y = compute_stddev(y_vals)
    if std_x > 0 and std_y > 0:
        return compute_covariance(x_vals, y_vals) / std_x / std_y
    return 0.0

# ---------- data scaling operations
def remove_mean(data: List[VectorType]) -> List[VectorType]:
    """Subtract column means from data."""
    col_means = np.mean(data, axis=0)
    return data - col_means

def get_scaling_params(data: List[VectorType]) -> Tuple[VectorType, VectorType]:
    """Return column means and standard deviations."""
    col_means = np.mean(data, axis=0)
    col_stds = np.std(data, axis=0, ddof=1)
    return col_means, col_stds

# Test scaling functionality
test_vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
test_means, test_stds = get_scaling_params(test_vectors)
assert np.all(test_means == [-1, 0, 1])
assert np.all(test_stds == [2, 1, 0])

def apply_scaling(data: List[VectorType],
                  ref_means: List[VectorType],
                  ref_stds: List[VectorType]) -> List[VectorType]:
    """Rescale data using reference parameters."""
    data_arr = np.array(data)
    rows, cols = np.shape(data_arr)
    scaled = data_arr.copy()
    for j in range(cols):
        if ref_stds[j] > 0:
            scaled[:, j] = (scaled[:, j] - ref_means[j]) / ref_stds[j]
    return scaled

def standardize(data: List[VectorType]) -> List[VectorType]:
    """Standardize data to zero mean and unit variance."""
    data_arr = np.array(data)
    rows, cols = np.shape(data_arr)
    col_means, col_stds = get_scaling_params(data_arr)
    standardized = data_arr.copy()
    for j in range(cols):
        if col_stds[j] > 0:
            standardized[:, j] = (standardized[:, j] - col_means[j]) / col_stds[j]
    return standardized

mean_check, std_check = get_scaling_params(standardize(test_vectors))
assert np.all(mean_check == [0, 0, 1])
assert np.all(std_check == [1, 1, 0])
assert np.all(standardize(test_vectors) == apply_scaling(test_vectors, test_means, test_stds))

def reverse_scaling(data: List[VectorType],
                    ref_means: List[VectorType],
                    ref_stds: List[VectorType]) -> List[VectorType]:
    """Revert standardized data back to original scale."""
    data_arr = np.array(data)
    rows, cols = np.shape(data_arr)
    restored = data_arr.copy()
    for j in range(cols):
        restored[:, j] = ref_means[j] + data_arr[:, j] * ref_stds[j]
    return restored

# Test reverse scaling
assert np.all(test_vectors == reverse_scaling(standardize(test_vectors), test_means, test_stds))

# ---------- data splitting utilities
def random_split(data: List[VectorType], split_prob: float) -> Tuple[List[VectorType], List[VectorType]]:
    """Split data into two sets with given probability."""
    random.seed(20062000)
    shuffled = data[:]
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * split_prob)
    return shuffled[:cutoff], shuffled[cutoff:]

def train_test_partition(features: List[VectorType],
                         labels: List[VectorType],
                         test_fraction: float) -> Tuple[List[VectorType], List[VectorType], List[VectorType], List[VectorType]]:
    """Partition data into training and testing sets."""
    indices = list(range(len(features)))
    train_idx, test_idx = random_split(indices, 1 - test_fraction)
    return ([features[i] for i in train_idx],
            [features[i] for i in test_idx],
            [labels[i] for i in train_idx],
            [labels[i] for i in test_idx])

def compute_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate classification accuracy."""
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

assert compute_accuracy(70, 4930, 13930, 981070) == 0.98114

def compute_precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate precision."""
    return tp / (tp + fp)

assert compute_precision(70, 4930, 13930, 981070) == 0.014

def compute_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate recall/sensitivity."""
    return tp / (tp + fn)

assert compute_recall(70, 4930, 13930, 981070) == 0.005

def compute_f1(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate F1 score."""
    p = compute_precision(tp, fp, fn, tn)
    r = compute_recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

assert compute_f1(70, 4930, 13930, 981070) == 0.00736842105263158