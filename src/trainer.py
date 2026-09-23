import os

import numpy as np

from .evaluator import evaluate_accuracy

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model.npz"
)


def train_val_split(X, y, val_fraction=0.1, seed=0):
    """
    Carves a validation set out of the training data.

    Early stopping has to be judged on data the final score is not reported
    on; selecting on the test set would make that score optimistic.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, patience=5,
                batch_size=128, save_path=None, seed=0):
    """
    Trains with mini-batch gradient descent and stops early once the
    validation loss has failed to improve for `patience` epochs.

    Batching matters here: a full-batch step over all 50k images is one
    update per epoch, so the network barely moves in a reasonable number
    of epochs.

    On return the model holds the best checkpoint, not the last epoch's
    weights, so whatever is evaluated afterwards is the model that was
    actually selected.

    Returns the path the best weights were written to.
    """
    save_path = save_path or DEFAULT_MODEL_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            Xb, yb = X_train[batch], y_train[batch]
            model.backward(Xb, yb, model.forward(Xb))

        val_probs = model.forward(X_val)
        val_loss = model.loss(y_val, val_probs)
        val_acc = evaluate_accuracy(y_val, np.argmax(val_probs, axis=1))

        print(f"Epoch {epoch + 1}/{epochs} - val loss: {float(val_loss):.4f} "
              f"- val accuracy: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            model.save(save_path)
            print(f"  improved - saved to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}. "
                      f"Best validation loss: {best_loss:.4f}")
                break

    # Restore the selected checkpoint; the loop may have moved past it.
    best = type(model).load(save_path)
    model.w1, model.b1 = best.w1, best.b1
    model.w2, model.b2 = best.w2, best.b2
    model.w3, model.b3 = best.w3, best.b3

    print(f"Training complete. Best model saved to {save_path}")
    return save_path
