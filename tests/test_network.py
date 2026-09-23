import numpy as np
import pytest

from src.helper_functions import loss, relu, relu_derivative, softmax
from src.neural_network import NeuralNetwork3Layer
from src.preprocessor import preprocess_data
from src.trainer import train_val_split


def test_softmax_rows_sum_to_one():
    out = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_softmax_is_stable_for_large_logits():
    # a naive exp() would overflow to inf/nan here
    out = softmax(np.array([[10_000.0, 10_001.0]]))
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(), 1.0)


def test_relu_and_derivative():
    x = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(relu(x), [[0.0, 0.0, 3.0]])
    assert np.array_equal(relu_derivative(x), [[0.0, 0.0, 1.0]])


def test_loss_is_lower_when_prediction_is_confident_and_correct():
    y = np.array([0])
    confident = np.array([[0.99, 0.01]])
    unsure = np.array([[0.51, 0.49]])
    assert loss(y, confident) < loss(y, unsure)


def test_preprocess_flattens_and_scales():
    x = np.full((4, 32, 32, 3), 255, dtype=np.uint8)
    a, b = preprocess_data(x, x)
    assert a.shape == (4, 3072)
    assert a.max() <= 1.0 and a.min() >= 0.0


def test_forward_returns_probabilities():
    net = NeuralNetwork3Layer(10, 3, 8, 4, seed=0)
    out = net.forward(np.random.default_rng(0).random((5, 10)))
    assert out.shape == (5, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_training_reduces_loss_on_a_separable_problem():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.1, (60, 6)), rng.normal(1, 0.1, (60, 6))])
    y = np.array([0] * 60 + [1] * 60)

    net = NeuralNetwork3Layer(6, 2, 16, 8, learning_rate=0.5, seed=0)
    before = net.loss(y, net.forward(X))
    for _ in range(200):
        net.backward(X, y, net.forward(X))
    after = net.loss(y, net.forward(X))

    assert after < before
    assert (net.predict(X) == y).mean() > 0.9


def test_save_load_round_trip(tmp_path):
    net = NeuralNetwork3Layer(10, 3, 8, 4, seed=1)
    X = np.random.default_rng(2).random((7, 10))
    path = tmp_path / "m.npz"
    net.save(str(path))

    restored = NeuralNetwork3Layer.load(str(path))
    assert np.array_equal(net.predict(X), restored.predict(X))


def test_train_val_split_is_disjoint_and_covers_everything():
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.arange(100)
    xt, yt, xv, yv = train_val_split(X, y, val_fraction=0.2, seed=0)
    assert len(yv) == 20 and len(yt) == 80
    assert set(yt).isdisjoint(set(yv))
    assert set(yt) | set(yv) == set(range(100))
