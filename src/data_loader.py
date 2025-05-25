"""
Module for data loading
"""

from keras.datasets import cifar10

def load_cifar10_data():
    """
    Loads the CIFAR-10 dataset in two tuples.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test