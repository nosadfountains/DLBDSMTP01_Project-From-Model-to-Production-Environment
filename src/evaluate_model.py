"""
Check a saved model (WP4 "done when").

Loads models/model_<version>.joblib in a fresh Python process, prepares the
test images again and compares the result with the numbers that
src/train.py stored in models/model_<version>.json.

If both match, the saved file is complete: model, classes and
preprocessing settings are enough to reproduce the results.

Run from the project folder:  python src/evaluate_model.py
Another version:              python src/evaluate_model.py v2
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import preprocessing

SPLIT_FILE = Path("data/split.csv")
MODELS_DIR = Path("models")


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    model_file = MODELS_DIR / f"model_{version}.joblib"
    metadata_file = MODELS_DIR / f"model_{version}.json"
    for needed in (SPLIT_FILE, model_file, metadata_file):
        if not needed.exists():
            print(f"File not found: {needed.resolve()}")
            print("Run this script from the project folder, after src/train.py.")
            return

    metadata = json.loads(metadata_file.read_text())
    model = joblib.load(model_file)
    print(f"Loaded model {version} ({metadata['model_type']}), "
          f"classes: {', '.join(metadata['classes'])}")

    # The preprocessing used now must be the one the model was trained with.
    if metadata["preprocessing"] != preprocessing.settings():
        print("WARNING: preprocessing settings differ from training!")
        print(f"  trained with: {metadata['preprocessing']}")
        print(f"  now:          {preprocessing.settings()}")

    table = pd.read_csv(SPLIT_FILE)
    test = table[table["split"] == "test"]
    X = np.stack([preprocessing.load_image_file(p) for p in test["path"]])
    probabilities = model.predict_proba(X)
    predicted = model.classes_[probabilities.argmax(axis=1)]

    now = {"accuracy": accuracy_score(test["label"], predicted),
           "macro_f1": f1_score(test["label"], predicted, average="macro")}
    saved = metadata["test_metrics"]

    print(f"\n{'Metric':<10}{'saved':>9}{'now':>9}")
    all_match = True
    for name in now:
        match = abs(now[name] - saved[name]) < 1e-9
        all_match = all_match and match
        print(f"{name:<10}{saved[name]:>9.3f}{now[name]:>9.3f}  {'OK' if match else 'DIFFERENT'}")

    print("\nResult: the saved model reproduces the test metrics." if all_match
          else "\nResult: the numbers differ - check preprocessing and versions.")


if __name__ == "__main__":
    main()