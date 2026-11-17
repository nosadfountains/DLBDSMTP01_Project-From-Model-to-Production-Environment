"""
Tests for the nightly batch job in src/batch_predict.py.

No real API is started: the batch job's calls are answered by the Flask
app from src/api.py with a tiny stand-in model, and all folders are
temporary test folders.

Run from the project folder:  python -m pytest
"""

import csv
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
import batch_predict  # noqa: E402
import preprocessing  # noqa: E402

api.LOG_FILE = Path(tempfile.gettempdir()) / "api_test.log"
CLASSES = ["dress", "pants", "shirt", "shoes", "shorts"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Temporary folders + a test API that the batch job talks to."""
    # tiny stand-in model
    rng = np.random.default_rng(0)
    X = rng.random((50, preprocessing.n_features()))
    y = [CLASSES[i % 5] for i in range(50)]
    joblib.dump(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y),
                tmp_path / "m.joblib")
    (tmp_path / "m.json").write_text(json.dumps({
        "version": "test", "model_type": "rf", "classes": CLASSES,
        "preprocessing": preprocessing.settings(),
        "test_metrics": {"accuracy": 0, "balanced_accuracy": 0, "macro_f1": 0}}))
    client = api.create_app(tmp_path / "m.joblib", tmp_path / "m.json").test_client()

    # temporary folders instead of the real ones
    for name, sub in [("INCOMING_DIR", "incoming"), ("PROCESSED_DIR", "processed"),
                      ("FAILED_DIR", "failed"), ("RESULTS_DIR", "results"),
                      ("LOG_DIR", "logs")]:
        monkeypatch.setattr(batch_predict, name, tmp_path / sub)
    (tmp_path / "incoming").mkdir()
    batch_predict.log.handlers.clear()  # log into the temporary folder

    # the batch job's API calls go to the test app
    def fake_send_chunk(paths):
        data = {"images": [(io.BytesIO(p.read_bytes()), p.name) for p in paths]}
        response = client.post("/predict", data=data)
        assert response.status_code == 200
        return response.get_json()
    monkeypatch.setattr(batch_predict, "send_chunk", fake_send_chunk)
    monkeypatch.setattr(batch_predict, "api_is_healthy", lambda: True)
    return tmp_path


def add_image(folder, name):
    Image.new("RGB", (100, 150), (30, 60, 200)).save(folder / name)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_images_are_predicted_and_moved(setup):
    for i in range(3):
        add_image(setup / "incoming", f"img{i}.png")
    summary = batch_predict.run_batch()

    assert summary["status"] == "ok" and summary["predicted"] == 3
    assert list((setup / "incoming").iterdir()) == []
    assert len(list((setup / "processed").iterdir())) == 3
    rows = read_csv(setup / "results" / summary["results_file"])
    assert [r["image_id"] for r in rows] == ["img0.png", "img1.png", "img2.png"]
    assert rows[0]["top_class"] in CLASSES


def test_corrupt_image_goes_to_failed(setup):
    add_image(setup / "incoming", "good.png")
    (setup / "incoming" / "broken.jpg").write_bytes(b"not an image")
    summary = batch_predict.run_batch()

    assert summary["predicted"] == 1 and summary["failed_images"] == 1
    assert (setup / "failed" / "broken.jpg").exists()
    assert (setup / "processed" / "good.png").exists()


def test_second_run_does_nothing_twice(setup):
    add_image(setup / "incoming", "a.png")
    first = batch_predict.run_batch()
    second = batch_predict.run_batch()

    assert first["predicted"] == 1
    assert second["images_found"] == 0 and second["predicted"] == 0
    assert len(read_csv(setup / "results" / first["results_file"])) == 1
    assert len(read_csv(setup / "logs" / "run_summary.csv")) == 2


def test_more_images_than_one_chunk(setup, monkeypatch):
    monkeypatch.setattr(batch_predict, "CHUNK_SIZE", 2)
    for i in range(5):
        add_image(setup / "incoming", f"img{i}.png")
    summary = batch_predict.run_batch()
    assert summary["predicted"] == 5


def test_api_down_leaves_images_waiting(setup, monkeypatch):
    monkeypatch.setattr(batch_predict, "api_is_healthy", lambda: False)
    add_image(setup / "incoming", "a.png")
    summary = batch_predict.run_batch()

    assert summary["status"] == "api_unavailable"
    assert summary["left_in_incoming"] == 1
    assert (setup / "incoming" / "a.png").exists()