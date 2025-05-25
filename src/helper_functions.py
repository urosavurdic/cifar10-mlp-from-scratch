import numpy as np
"""
This file contains helper functions that could be used for other NNs
later like CNN that gives better results for image classification
"""

def softmax(x):
    """
    Used for output layer since it is multiclass classification
    This form ensures numerical stability
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    """
    Used for hidden layers to avoid vanishing
    """
    return np.maximum(0, x)


def relu_derivate(x):
    return (x > 0).astype(float)


def loss(y_true, y_pred, eps=1e-15):
    n = len(y_true)
    probs = np.clip(y_pred[np.arange(n), y_true], eps, 1 - eps)
    return -np.mean(np.log(probs))

def loss2(y_true, y_pred):
    """
    Calculates the categorical cross-entropy loss (memory-efficient version).
    """
    m = y_true.shape[0]
    eps = 1e-15
    total_loss = 0.0

    for i in range(m):
        prob = min(max(y_pred[i, y_true[i]], eps), 1 - eps)
        total_loss += -np.log(prob)

    return total_loss / m
