# CIFAR-10 MLP from scratch

A three-layer fully connected neural network for CIFAR-10, written in NumPy.
Forward pass, backpropagation, softmax cross-entropy and mini-batch SGD are all
implemented by hand — no autograd, and no deep learning framework anywhere in
the dependency list. A small Flask service serves the trained weights.

![Confusion matrix](models/confusion_matrix.png)

## What it does

- Implements a 3072 → 256 → 128 → 10 network with relu hidden layers and a
  softmax output, with gradients derived and coded directly
- Trains with mini-batch SGD and early stopping on a held-out validation split
- Reports accuracy, a per-class precision/recall/F1 table and a confusion matrix
- Serves predictions over HTTP from saved weights
- Downloads and parses the CIFAR-10 archive with the standard library

## Quick start

```bash
pip install -r requirements.txt
python -m src.main          # downloads CIFAR-10 on first run, then trains
```

Serve the trained weights:

```bash
python -m api.app
python api/test_request.py  # posts one random image
```

```bash
pytest tests/
```

## How it works

`src/main.py` loads CIFAR-10, scales pixels to `[0, 1]` and flattens each
32×32×3 image into a 3072-vector. A validation split is carved out of the
training set, and training stops once validation loss has not improved for five
epochs. The test set is touched exactly once, after model selection is finished,
so the reported accuracy is not inflated by early stopping.

Weights are stored as a `.npz` archive of plain arrays rather than a pickle, so
loading them cannot execute code.

| Path | What it is |
|---|---|
| `src/neural_network.py` | the network: forward, backward, save/load |
| `src/helper_functions.py` | softmax, relu and cross-entropy primitives |
| `src/trainer.py` | mini-batch loop, validation split, early stopping |
| `src/evaluator.py` | accuracy, classification report, confusion matrix |
| `src/data_loader.py` | fetches and unpacks the CIFAR-10 archive |
| `api/app.py` | Flask service: `/health`, `/predict` |
| `tests/` | unit tests for the network, training and the API |

## Results

Trained on the full 50,000-image training set (45,000 train / 5,000 validation),
stopped early at epoch 26, and evaluated once on the 10,000-image test set.

**Test accuracy: 48.3%** — against 10% for chance.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 0.57 | 0.49 | 0.53 |
| automobile | 0.58 | 0.61 | 0.59 |
| bird | 0.34 | 0.38 | 0.36 |
| cat | 0.38 | 0.28 | 0.32 |
| deer | 0.40 | 0.42 | 0.41 |
| dog | 0.42 | 0.32 | 0.37 |
| frog | 0.50 | 0.55 | 0.52 |
| horse | 0.53 | 0.54 | 0.54 |
| ship | 0.56 | 0.65 | 0.60 |
| truck | 0.52 | 0.57 | 0.54 |

The ceiling here is the architecture, not the training. A fully connected net
flattens the image and throws away the fact that neighbouring pixels are
related, so it has to learn every spatial pattern independently at every
position. The per-class split shows it: rigid, distinctly-coloured classes
(ship, automobile) score well, while the deformable animal classes it has to
recognise in many poses (cat at 0.28 recall, dog at 0.32) are where it fails.
That gap is what convolution exists to close.

## License

MIT — see [LICENSE](LICENSE).
