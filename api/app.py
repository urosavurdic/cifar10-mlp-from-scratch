"""
Minimal prediction service for the trained network.

Run from the repository root:

    python -m api.app

The weights path can be overridden with MODEL_PATH.
"""
import os

import numpy as np
from flask import Flask, jsonify, request

from src.evaluator import CLASS_NAMES
from src.neural_network import NeuralNetwork3Layer

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "model.npz"),
)

app = Flask(__name__)
_model = None


def get_model():
    """Loads the weights on first use so import does not require them."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No weights at {MODEL_PATH}. Train first with `python -m src.main`, "
                f"or point MODEL_PATH at an existing .npz."
            )
        _model = NeuralNetwork3Layer.load(MODEL_PATH)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts {"image": ...} as either a 32x32x3 nested list of 0-255 values
    or an already-flat 3072-vector, and returns the predicted class.
    """
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "request body must contain an 'image' field"}), 400

    img = np.asarray(data["image"], dtype=np.float32)

    if img.shape == (32, 32, 3):
        img = img.reshape(1, -1)
    elif img.shape == (3072,):
        img = img.reshape(1, -1)
    else:
        return jsonify({
            "error": f"expected shape (32, 32, 3) or (3072,), got {list(img.shape)}"
        }), 400

    # The network was trained on pixels scaled to [0, 1]; scale here only if
    # the caller sent raw 0-255 values, so a pre-scaled vector is not halved.
    if img.max() > 1.0:
        img = img / 255.0

    index = int(get_model().predict(img)[0])
    return jsonify({"prediction": index, "label": CLASS_NAMES[index]})


if __name__ == "__main__":
    app.run(port=5000)
