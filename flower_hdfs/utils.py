import numpy as np
import torch


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_class_weights(dataset, label_column="label", num_classes=None):
    labels = np.array(dataset[label_column])
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = np.where(counts == 0, 0.0, weights)
    return torch.tensor(weights, dtype=torch.float32)
