import numpy as np

from . import helper_functions as hf


class NeuralNetwork3Layer:
    """
    Fully connected 3072 -> h1 -> h2 -> 10 network with relu hidden layers
    and a softmax output, trained by hand-written backpropagation.
    """

    def __init__(self, input_size, output_size, hidden_size1, hidden_size2,
                 learning_rate=0.01, seed=None):
        """
        Args:
            input_size: length of a flattened input vector
            output_size: number of classes
            hidden_size1: width of the first hidden layer
            hidden_size2: width of the second hidden layer
            learning_rate: SGD step size
            seed: fixes the weight initialisation so runs are comparable

        Weights use He-style scaling (sqrt(1/fan_in)), which keeps the
        activation variance roughly constant across layers instead of
        letting it explode or decay with depth.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        self.w1 = rng.standard_normal((input_size, hidden_size1)) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.w2 = rng.standard_normal((hidden_size1, hidden_size2)) * np.sqrt(1.0 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.w3 = rng.standard_normal((hidden_size2, output_size)) * np.sqrt(1.0 / hidden_size2)
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        """Returns class probabilities for a batch of flattened images."""
        self.z1 = X @ self.w1 + self.b1
        self.a1 = hf.relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = hf.relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = hf.softmax(self.z3)

        return self.a3

    def backward(self, X, y_true, y_pred):
        """
        One gradient step over the batch.

        For softmax with cross-entropy the output-layer gradient collapses
        to (prediction - one_hot), which is why no softmax derivative is
        computed explicitly here.
        """
        m = y_true.shape[0]

        y_one_hot = np.zeros_like(self.a3)
        y_one_hot[np.arange(m), y_true] = 1

        dz3 = (self.a3 - y_one_hot) / m
        dw3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        dz2 = (dz3 @ self.w3.T) * hf.relu_derivative(self.z2)
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ self.w2.T) * hf.relu_derivative(self.z1)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        """Returns the argmax class index for each row of X."""
        return np.argmax(self.forward(X), axis=1)

    def loss(self, y_true, y_pred):
        return hf.loss(y_true, y_pred)

    def save(self, path):
        """
        Writes the weights to a .npz archive.

        Arrays rather than a pickle, so loading the file cannot execute
        code and the result is portable across Python versions.
        """
        np.savez(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3,
            learning_rate=self.learning_rate,
        )

    @classmethod
    def load(cls, path):
        """Rebuilds a network from a .npz archive written by save()."""
        d = np.load(path)
        model = cls(
            input_size=d["w1"].shape[0],
            output_size=d["w3"].shape[1],
            hidden_size1=d["w1"].shape[1],
            hidden_size2=d["w2"].shape[1],
            learning_rate=float(d["learning_rate"]),
        )
        model.w1, model.b1 = d["w1"], d["b1"]
        model.w2, model.b2 = d["w2"], d["b2"]
        model.w3, model.b3 = d["w3"], d["b3"]
        return model
