def preprocess_data(x_train, x_test):
    """
    Scales pixels to [0, 1] and flattens each image into a single vector.

    Args:
        x_train, x_test: arrays of shape (n_samples, height, width, channels)

    Returns:
        Arrays of shape (n_samples, height * width * channels). All three
        colour channels are kept.
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, x_test
