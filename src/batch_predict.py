"""
Nightly batch prediction (WP6).

Takes every image waiting in data/incoming/, sends them to the API in
chunks, and saves the results:

  data/incoming/   new refund photos wait here
  data/processed/  images that were predicted are moved here
  data/failed/     images the API could not read are moved here
  results/predictions_<date>.csv   one row per predicted image
  logs/batch.log                   what happened, line by line
  logs/run_summary.csv             one row per run (for monitoring, WP7)

Safe to run again at any time: images that were already handled are no
longer in data/incoming/, so nothing is predicted twice. If the API is
down or a chunk fails, the remaining images simply stay in data/incoming/
and the next run picks them up.

The API must be running (python src/api.py). For the nightly run,
run_nightly.sh starts the API, runs this script and stops the API again.

Run from the project folder:  python src/batch_predict.py
"""

import csv
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent  # works from any folder (cron)
INCOMING_DIR = PROJECT_DIR / "data" / "incoming"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FAILED_DIR = PROJECT_DIR / "data" / "failed"
RESULTS_DIR = PROJECT_DIR / "results"
LOG_DIR = PROJECT_DIR / "logs"

API_URL = "http://127.0.0.1:8000"
CHUNK_SIZE = 64        # the API accepts at most 64 images per request
REQUEST_TIMEOUT = 120  # seconds to wait for one chunk

log = logging.getLogger("batch")


def setup_logging():
    if log.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "batch.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def find_new_images():
    """All files waiting in data/incoming/, oldest name first."""
    if not INCOMING_DIR.exists():
        return []
    return sorted(p for p in INCOMING_DIR.iterdir()
                  if p.is_file() and not p.name.startswith("."))


def api_is_healthy():
    try:
        return requests.get(f"{API_URL}/health", timeout=10).status_code == 200
    except requests.RequestException:
        return False


def send_chunk(paths):
    """Send some images to the API in one request; return its JSON answer."""
    opened = [open(p, "rb") for p in paths]
    try:
        files = [("images", (p.name, f)) for p, f in zip(paths, opened)]
        response = requests.post(f"{API_URL}/predict", files=files, timeout=REQUEST_TIMEOUT)
    finally:
        for f in opened:
            f.close()
    response.raise_for_status()  # turns a 4xx/5xx answer into an error
    return response.json()


def save_results(rows, results_file):
    """Append rows to today's results CSV (header only for a new file)."""
    new_file = not results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def write_summary(summary):
    """Append one row per run to logs/run_summary.csv."""
    summary_file = LOG_DIR / "run_summary.csv"
    new_file = not summary_file.exists()
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)


def run_batch():
    """Process all waiting images once. Returns the run summary (a dict)."""
    setup_logging()
    for folder in (INCOMING_DIR, PROCESSED_DIR, FAILED_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    results_file = RESULTS_DIR / f"predictions_{started:%Y-%m-%d}.csv"
    summary = {"started": started.isoformat(timespec="seconds"), "status": "ok",
               "images_found": 0, "predicted": 0, "failed_images": 0,
               "left_in_incoming": 0, "model_version": "", "duration_s": 0,
               "results_file": results_file.name}
    start = time.perf_counter()

    images = find_new_images()
    summary["images_found"] = len(images)
    log.info("Run started: %d new images in %s", len(images), INCOMING_DIR)

    if images and not api_is_healthy():
        log.error("API not reachable or unhealthy at %s - nothing processed", API_URL)
        summary["status"] = "api_unavailable"
    else:
        for i in range(0, len(images), CHUNK_SIZE):
            chunk = images[i:i + CHUNK_SIZE]
            try:
                answer = send_chunk(chunk)
            except Exception as error:
                log.error("Chunk %d failed (%s) - stopping; remaining images stay "
                          "in incoming for the next run", i // CHUNK_SIZE + 1, error)
                summary["status"] = "chunk_failed"
                break

            summary["model_version"] = answer["model_version"]
            rows = []
            # The API answers in the same order the images were sent
            for path, item in zip(chunk, answer["predictions"]):
                if item["status"] == "ok":
                    rows.append({
                        "image_id": path.name,
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "model_version": answer["model_version"],
                        "top_class": item["top_class"],
                        "confidence": item["confidence"],
                        **{f"p_{c}": p for c, p in item["probabilities"].items()},
                    })
            if rows:
                save_results(rows, results_file)  # save first, then move

            for path, item in zip(chunk, answer["predictions"]):
                if item["status"] == "ok":
                    shutil.move(str(path), PROCESSED_DIR / path.name)
                    summary["predicted"] += 1
                else:
                    shutil.move(str(path), FAILED_DIR / path.name)
                    summary["failed_images"] += 1
                    log.warning("Unreadable image moved to failed/: %s (%s)",
                                path.name, item["error"])

    summary["left_in_incoming"] = len(find_new_images())
    summary["duration_s"] = round(time.perf_counter() - start, 1)
    write_summary(summary)
    log.info("Run finished: status=%s predicted=%d failed=%d left=%d (%.1f s)",
             summary["status"], summary["predicted"], summary["failed_images"],
             summary["left_in_incoming"], summary["duration_s"])
    return summary


if __name__ == "__main__":
    result = run_batch()
    sys.exit(0 if result["status"] == "ok" else 1)  # non-zero = problem (for cron)