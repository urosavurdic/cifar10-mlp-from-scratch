import numpy as np


def preprocess_data(x_train, x_test):
    """
    Normalize pixel values and flatten images

    Args:
        Training and test images, shape (num_samples, height, width, channels)

    Returns:
        Flattened and normalized training and test data (num_samples, height*width*channels)
    """
    # Convert to float and normalize pixel values between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten each image to a 1D vector (preserving all RGB channels)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, x_test
