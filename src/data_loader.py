"""
CIFAR-10 loading, straight from the canonical archive.

The point of this project is that nothing is imported to do the work, so
the data layer does not pull in a deep learning framework either. The
archive is fetched once into data/ and unpacked with the standard library.
"""
import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_EXTRACTED = "cifar-10-batches-py"


def _download(data_dir):
    """Fetches and unpacks the archive unless it is already present."""
    os.makedirs(data_dir, exist_ok=True)
    root = os.path.join(data_dir, _EXTRACTED)
    if os.path.isdir(root):
        return root

    archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print(f"Downloading CIFAR-10 from {URL} ...")
        urllib.request.urlretrieve(URL, archive)

    print("Extracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    return root


def _load_batch(path):
    """Reads one batch file into an image array and a label array."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    # stored flat as 3072 = 3x32x32 in channel-first order
    data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, np.array(batch[b"labels"])


def load_cifar10_data(data_dir=None):
    """
    Returns (x_train, y_train, x_test, y_test) as uint8 arrays of shape
    (n, 32, 32, 3) and (n,).
    """
    root = _download(data_dir or DEFAULT_DATA_DIR)

    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(os.path.join(root, f"data_batch_{i}"))
        xs.append(x)
        ys.append(y)
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    x_test, y_test = _load_batch(os.path.join(root, "test_batch"))

    return x_train, y_train, x_test, y_test
