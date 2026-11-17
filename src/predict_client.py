"""
Send a few images to the running API and print the answer (WP5 test client).

Start the API first, in its own terminal:  python src/api.py
Then, in a second terminal:

  python src/predict_client.py                 5 random images from the test set
  python src/predict_client.py a.jpg b.png     your own images

The batch job (WP6) will send images to the API in the same way.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"
SPLIT_FILE = Path("data/split.csv")


def pick_test_images(n=5):
    """Return n random image paths from the test split."""
    table = pd.read_csv(SPLIT_FILE)
    return list(table[table["split"] == "test"].sample(n)["path"])


def main():
    paths = sys.argv[1:] or pick_test_images()
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        print(f"File not found: {', '.join(missing)}")
        return

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
    except requests.ConnectionError:
        print(f"No API found at {API_URL}. Start it first: python src/api.py")
        return
    print(f"Health: {health.status_code} {health.json()}\n")

    # One request with all images, each under the field name "images"
    opened = [open(p, "rb") for p in paths]
    try:
        files = [("images", (Path(p).name, f)) for p, f in zip(paths, opened)]
        response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    finally:
        for f in opened:
            f.close()

    print(f"Status: {response.status_code}")
    answer = response.json()
    if response.status_code != 200:
        print(f"Error: {answer.get('error')}")
        return

    print(f"Model {answer['model_version']}: {answer['n_images']} images, "
          f"{answer['n_failed']} failed\n")
    for path, item in zip(paths, answer["predictions"]):
        if item["status"] == "ok":
            print(f"  {path:<45} -> {item['top_class']:<7} ({item['confidence']:.0%})")
        else:
            print(f"  {path:<45} -> ERROR: {item['error']}")


if __name__ == "__main__":
    main()