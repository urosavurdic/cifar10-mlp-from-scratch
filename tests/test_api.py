import numpy as np
import pytest

from src.neural_network import NeuralNetwork3Layer

flask = pytest.importorskip("flask")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Serves a freshly initialised network so the tests need no training run."""
    weights = tmp_path / "model.npz"
    NeuralNetwork3Layer(3072, 10, 16, 8, seed=0).save(str(weights))
    monkeypatch.setenv("MODEL_PATH", str(weights))

    import importlib

    from api import app as app_module
    importlib.reload(app_module)
    return app_module.app.test_client()


def test_health_reports_ok(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_accepts_a_32x32x3_image(client):
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8).tolist()
    body = client.post("/predict", json={"image": img}).get_json()
    assert 0 <= body["prediction"] <= 9
    assert isinstance(body["label"], str)


def test_accepts_an_already_flat_vector(client):
    body = client.post("/predict", json={"image": np.random.random(3072).tolist()}).get_json()
    assert 0 <= body["prediction"] <= 9


def test_rejects_the_wrong_shape(client):
    r = client.post("/predict", json={"image": [[1, 2], [3, 4]]})
    assert r.status_code == 400
    assert "expected shape" in r.get_json()["error"]


def test_rejects_a_missing_image_field(client):
    assert client.post("/predict", json={}).status_code == 400
