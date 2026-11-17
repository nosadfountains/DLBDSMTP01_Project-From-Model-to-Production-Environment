"""
The model as a REST service (WP5), built with Flask.

Endpoints:
  GET  /health       is the service ready? (200 = yes, 503 = no model loaded)
  GET  /model-info   model version, classes, preprocessing settings, test metrics
  POST /predict      one or more images in, class probabilities out

How to send images to /predict:
  as a normal file upload ("multipart/form-data"), every image under the
  field name "images". Example with curl (from the project folder):

    curl -F "images=@data/raw/black_dress/00000001.jpg" http://127.0.0.1:8000/predict

  The answer is JSON with one entry per image, in the order they were sent:
    {"id": "00000001.jpg", "status": "ok", "top_class": "dress",
     "confidence": 0.81, "probabilities": {"dress": 0.81, "pants": 0.02, ...}}
  An image that cannot be read gets "status": "error" and a message;
  the other images in the same request are still predicted.

Start the service from the project folder:  python src/api.py
Stop it with Ctrl+C. Every request is written to logs/api.log.
"""

import json
import logging
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import preprocessing  # our own file in src/

PROJECT_DIR = Path(__file__).resolve().parent.parent  # works from any folder
MODEL_VERSION = "v1"
MODEL_FILE = PROJECT_DIR / "models" / f"model_{MODEL_VERSION}.joblib"
METADATA_FILE = PROJECT_DIR / "models" / f"model_{MODEL_VERSION}.json"
LOG_FILE = PROJECT_DIR / "logs" / "api.log"

MAX_BATCH_SIZE = 64             # most images per request; the batch job sends chunks
MAX_REQUEST_MB = 50             # bigger requests are refused before they are read
HOST, PORT = "127.0.0.1", 8000  # not 5000: on macOS that port is used by AirPlay

log = logging.getLogger("api")


def setup_logging():
    """Write log lines to logs/api.log and to the terminal."""
    if log.handlers:  # already set up
        return
    LOG_FILE.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def load_model(model_file, metadata_file):
    """Return (model, metadata), or (None, None) if they cannot be used."""
    try:
        model = joblib.load(model_file)
        metadata = json.loads(Path(metadata_file).read_text())
    except Exception as error:
        log.error("Model could not be loaded from %s: %s", model_file, error)
        return None, None

    # Training and API must prepare images the same way (see preprocessing.py)
    if metadata.get("preprocessing") != preprocessing.settings():
        log.error("Preprocessing settings differ from training: %s vs %s",
                  metadata.get("preprocessing"), preprocessing.settings())
        return None, None

    log.info("Model %s loaded (%s)", metadata["version"], metadata["model_type"])
    return model, metadata


def create_app(model_file=MODEL_FILE, metadata_file=METADATA_FILE):
    """Build the Flask app. The tests call this with a small test model."""
    setup_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_MB * 1024 * 1024
    model, metadata = load_model(model_file, metadata_file)

    def error(status, message):
        """A short JSON error message for the client."""
        return jsonify({"error": message}), status

    @app.get("/health")
    def health():
        if model is None:
            return jsonify({"status": "unhealthy", "reason": "no model loaded"}), 503
        return jsonify({"status": "ok", "model_version": metadata["version"]})

    @app.get("/model-info")
    def model_info():
        if model is None:
            return error(503, "No model loaded.")
        return jsonify({
            "version": metadata["version"],
            "model_type": metadata["model_type"],
            "classes": metadata["classes"],
            "preprocessing": metadata["preprocessing"],
            "test_metrics": {k: metadata["test_metrics"][k]
                             for k in ("accuracy", "balanced_accuracy", "macro_f1")},
            "max_batch_size": MAX_BATCH_SIZE,
        })

    @app.post("/predict")
    def predict():
        start = time.perf_counter()
        if model is None:
            return error(503, "No model loaded. Train it first: python src/train.py")

        files = request.files.getlist("images")
        if not files:
            return error(400, "No images found. Send them as a file upload "
                              "with the field name 'images'.")
        if len(files) > MAX_BATCH_SIZE:
            return error(413, f"Too many images: {len(files)}. "
                              f"Send at most {MAX_BATCH_SIZE} per request.")

        # Prepare every image; a broken one is marked, the others go on
        results, features, ok_positions = [], [], []
        for position, file in enumerate(files):
            image_id = file.filename or f"image_{position}"
            try:
                features.append(preprocessing.load_image_bytes(file.read()))
                ok_positions.append(position)
                results.append({"id": image_id, "status": "ok"})
            except Exception as err:
                results.append({"id": image_id, "status": "error",
                                "error": f"not a readable image ({type(err).__name__})"})

        # Predict all good images in one go
        if features:
            probabilities = model.predict_proba(features)
            for position, probs in zip(ok_positions, probabilities):
                best = probs.argmax()
                results[position].update({
                    "top_class": str(model.classes_[best]),
                    "confidence": round(float(probs[best]), 4),
                    "probabilities": {str(c): round(float(p), 4)
                                      for c, p in zip(model.classes_, probs)},
                })

        n_failed = len(files) - len(features)
        ms = (time.perf_counter() - start) * 1000
        log.info("predict images=%d ok=%d failed=%d duration_ms=%.0f",
                 len(files), len(features), n_failed, ms)
        return jsonify({
            "model_version": metadata["version"],
            "n_images": len(files),
            "n_failed": n_failed,
            "predictions": results,
        })

    @app.errorhandler(HTTPException)
    def http_error(err):
        # e.g. 404 wrong address, 405 wrong method, 413 request too large
        log.warning("%s %s -> %s", request.method, request.path, err.code)
        return error(err.code, err.description)

    @app.errorhandler(Exception)
    def unexpected_error(err):
        # Full details go into the log; the client only gets a short message
        log.exception("Unexpected error on %s %s", request.method, request.path)
        return error(500, "Internal error. Details are in logs/api.log.")

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\nAPI running on http://{HOST}:{PORT}  (stop with Ctrl+C)\n")
    app.run(host=HOST, port=PORT)