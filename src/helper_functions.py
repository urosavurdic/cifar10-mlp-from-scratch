"""
Activation and loss primitives shared by the network.

Kept separate from the model so a different architecture (a CNN, say) can
reuse them unchanged.
"""
import numpy as np


def softmax(x):
    """
    Output-layer activation for multiclass classification.

    Subtracting the row max before exponentiating keeps this stable for
    large logits, where a plain exp would overflow.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    """Hidden-layer activation; avoids the vanishing gradients of a sigmoid."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Sub-gradient of relu, taking the derivative at 0 to be 0."""
    return (x > 0).astype(float)


def loss(y_true, y_pred, eps=1e-15):
    """
    Categorical cross-entropy over integer labels.

    Only the predicted probability of the true class contributes, so this
    indexes those directly instead of building a one-hot matrix. Clipping
    keeps log() away from 0.
    """
    n = len(y_true)
    probs = np.clip(y_pred[np.arange(n), y_true], eps, 1 - eps)
    return -np.mean(np.log(probs))
