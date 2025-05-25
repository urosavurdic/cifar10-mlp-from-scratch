import numpy as np
import helper_functions as hf

class NeuralNetwork3Layer:
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, learning_rate=0.01):
        """
        Args:
            input_size: size of input data
            output_size: size of output data (effectively number of classes)
            hidden_size1: size of hidden layer 1
            hidden_size2: size of hidden layer 2
            learning_rate: how fast nn learns

        Functionality:
            Constructor makes initialization of weights and biases at the beginning of training
            For better stability (avoiding vanishing and exploding) Xavier/Glorot-style is used
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(input_size, hidden_size1) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.w2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(1. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.w3 = np.random.randn(hidden_size2, output_size) * np.sqrt(1. / hidden_size2)
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Forward pass of the neural network
        """
        # layer 1
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = hf.relu(self.z1)

        # layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = hf.relu(self.z2)

        # layer 3
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = hf.softmax(self.z3)

        return self.a3

    def backward(self, X, y_true, y_pred):
        """
        Performs backward propagation of the neural network
        """
        m = y_true.shape[0]

        # one-hot encode labels
        y_one_hot = np.zeros_like(self.a3)
        y_one_hot[np.arange(m), y_true] = 1

        # output layer gradients
        dz3 = (self.a3 - y_one_hot) / m
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # layer 2 gradients
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * hf.relu_derivate(self.z2)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # layer 1 gradients
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * hf.relu_derivate(self.z1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # parameter update
        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3

        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def loss(self, y_true, y_pred):
        return hf.loss(y_true, y_pred)