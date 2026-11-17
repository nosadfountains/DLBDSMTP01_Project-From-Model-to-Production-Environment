"""
Tests for the API in src/api.py.

The real model is not on GitHub, so each test run trains a tiny stand-in
model on random pixels. The tests check how the API behaves (answers,
error messages), not how good the model is.

Run from the project folder:  python -m pytest
"""

import io
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import api  # noqa: E402
import preprocessing  # noqa: E402

# Test requests go into a temporary log, not into the real logs/api.log
api.LOG_FILE = Path(tempfile.gettempdir()) / "api_test.log"

CLASSES = ["dress", "pants", "shirt", "shoes", "shorts"]


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    """Train a tiny model and save it like src/train.py does."""
    folder = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(0)
    X = rng.random((50, preprocessing.n_features()))
    y = [CLASSES[i % len(CLASSES)] for i in range(50)]
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    joblib.dump(model, folder / "model.joblib")
    metadata = {
        "version": "test", "model_type": "random_forest", "classes": CLASSES,
        "preprocessing": preprocessing.settings(),
        "test_metrics": {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5},
    }
    (folder / "model.json").write_text(json.dumps(metadata))
    return folder / "model.joblib", folder / "model.json", model


@pytest.fixture
def client(model_files):
    app = api.create_app(model_files[0], model_files[1])
    return app.test_client()


def png_bytes(width=120, height=200, colour=(30, 60, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(*items):
    """Build a multipart upload: items are (file name, bytes)."""
    return {"images": [(io.BytesIO(data), name) for name, data in items]}


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_version"] == "test"


def test_predict_valid_batch(client):
    response = client.post("/predict", data=upload(("a.png", png_bytes()),
                                                   ("b.png", png_bytes(300, 150))))
    assert response.status_code == 200
    answer = response.get_json()
    assert answer["n_images"] == 2 and answer["n_failed"] == 0
    for item in answer["predictions"]:
        assert item["status"] == "ok"
        assert set(item["probabilities"]) == set(CLASSES)
        assert abs(sum(item["probabilities"].values()) - 1) < 0.01
        assert item["top_class"] in CLASSES


def test_api_matches_direct_prediction(client, model_files):
    """The API must give the same probabilities as the model used directly."""
    data = png_bytes(150, 400, (200, 30, 30))
    item = client.post("/predict", data=upload(("x.png", data))).get_json()["predictions"][0]
    model = model_files[2]
    expected = model.predict_proba([preprocessing.load_image_bytes(data)])[0]
    for cls, p in zip(model.classes_, expected):
        assert abs(item["probabilities"][cls] - p) < 0.001


def test_corrupt_image_is_marked_others_still_work(client):
    response = client.post("/predict", data=upload(("good.png", png_bytes()),
                                                   ("broken.jpg", b"not an image"),
                                                   ("good2.png", png_bytes())))
    assert response.status_code == 200
    answer = response.get_json()
    assert answer["n_failed"] == 1
    statuses = [p["status"] for p in answer["predictions"]]
    assert statuses == ["ok", "error", "ok"]   # same order as sent
    assert answer["predictions"][1]["id"] == "broken.jpg"


def test_empty_request(client):
    response = client.post("/predict")
    assert response.status_code == 400
    assert "images" in response.get_json()["error"]


def test_too_many_images(client):
    items = [(f"{i}.png", png_bytes(20, 20)) for i in range(api.MAX_BATCH_SIZE + 1)]
    response = client.post("/predict", data=upload(*items))
    assert response.status_code == 413


def test_no_model_gives_503(tmp_path):
    app = api.create_app(tmp_path / "missing.joblib", tmp_path / "missing.json")
    client = app.test_client()
    assert client.get("/health").status_code == 503
    response = client.post("/predict", data=upload(("a.png", png_bytes())))
    assert response.status_code == 503


def test_wrong_address_gives_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()