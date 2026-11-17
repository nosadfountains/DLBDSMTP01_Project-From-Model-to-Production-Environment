"""
Train the image classifier (WP4).

What the script does:
  1. reads data/split.csv (written by src/split_data.py)
  2. prepares all train and validation images with src/preprocessing.py
  3. trains a "dummy" baseline and a few simple scikit-learn models
  4. compares them on the validation images and picks the best
  5. retrains the best one on train + validation
  6. evaluates it ONCE on the test images (accuracy, precision, recall, F1,
     confusion matrix)
  7. saves the model, its metadata and the results

Files written:
  models/model_<version>.joblib      the trained model (not on GitHub)
  models/model_<version>.json        classes, preprocessing settings, metrics
  docs/model_report_<version>.txt    readable results for the project log/slides
  docs/confusion_matrix_<version>.png

Run from the project folder:  python src/train.py
Takes a few minutes; most of the time goes into reading the images.
"""

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")  # draw charts into files, no window needed
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

import preprocessing  # noqa: E402  (our own file in src/)

SPLIT_FILE = Path("data/split.csv")
MODELS_DIR = Path("models")
DOCS_DIR = Path("docs")

MODEL_VERSION = "v1"          # raise to v2, v3 ... when you retrain with changes
RANDOM_SEED = 42              # same results every time the script runs
REFIT_ON_TRAIN_AND_VAL = True  # retrain the winner on train + validation before the test


def candidate_models():
    """The models compared on the validation images.

    class_weight="balanced" makes mistakes on rare classes (shorts) count more,
    so the model does not simply favour the big classes (shoes).
    StandardScaler + PCA first squeeze the 4,096 pixel values into 100 numbers
    that capture the main shapes; this makes training much faster.
    """
    return {
        "pca_logistic_regression": make_pipeline(
            StandardScaler(),
            PCA(n_components=100, random_state=RANDOM_SEED),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "pca_svm": make_pipeline(
            StandardScaler(),
            PCA(n_components=100, random_state=RANDOM_SEED),
            # SVC gives no probabilities by itself; CalibratedClassifierCV adds them
            CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced"), ensemble=False),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED),
    }


def load_split(name, table):
    """Prepare all images of one split. Returns (features X, labels y)."""
    rows = table[table["split"] == name]
    features = np.empty((len(rows), preprocessing.n_features()), dtype=np.float32)
    for i, path in enumerate(rows["path"]):
        try:
            features[i] = preprocessing.load_image_file(path)
        except Exception as error:
            sys.exit(f"Could not read {path}: {type(error).__name__}: {error}")
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1:>5} / {len(rows)} images")
    print(f"  {name:<6} {len(rows):>5} images ready")
    return features, rows["label"].to_numpy()


def compare_on_validation(X_train, y_train, X_val, y_val):
    """Train every candidate on train, score it on validation. Returns a table."""
    models = {"dummy_most_frequent": DummyClassifier(strategy="most_frequent")}
    models.update(candidate_models())

    results = []
    for name, model in models.items():
        print(f"  training {name} ...", flush=True)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        predicted = model.predict(X_val)
        results.append({
            "model": name,
            "val_accuracy": accuracy_score(y_val, predicted),
            "val_macro_f1": f1_score(y_val, predicted, average="macro"),
            "train_seconds": seconds,
        })
    return pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)


def top_class(model, X):
    """Class with the highest probability - the same rule the API will use."""
    probabilities = model.predict_proba(X)
    return model.classes_[probabilities.argmax(axis=1)]


def save_confusion_matrix(y_true, y_pred, classes, path):
    """Save the confusion matrix as a picture for the slides."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=classes, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix, test set (model {MODEL_VERSION})")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    if not SPLIT_FILE.exists():
        print(f"File not found: {SPLIT_FILE.resolve()}")
        print("Run this script from the project folder, after src/split_data.py.")
        return
    MODELS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    table = pd.read_csv(SPLIT_FILE)

    print("Step 1/4  Preparing images "
          f"({preprocessing.IMAGE_SIZE}x{preprocessing.IMAGE_SIZE}, "
          f"grayscale={preprocessing.GRAYSCALE})")
    X_train, y_train = load_split("train", table)
    X_val, y_val = load_split("val", table)

    print("\nStep 2/4  Comparing models on the validation images")
    comparison = compare_on_validation(X_train, y_train, X_val, y_val)
    print()
    print(comparison.to_string(index=False, float_format="%.3f"))
    best_name = comparison.iloc[0]["model"]
    print(f"\n  Best model (highest macro F1 on validation): {best_name}")

    print("\nStep 3/4  Training the final model")
    model = candidate_models()[best_name]
    if REFIT_ON_TRAIN_AND_VAL:
        X_fit = np.concatenate([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
        print(f"  on train + validation ({len(y_fit)} images)")
    else:
        X_fit, y_fit = X_train, y_train
        print(f"  on train only ({len(y_fit)} images)")
    model.fit(X_fit, y_fit)

    print("\nStep 4/4  Evaluating once on the test images")
    X_test, y_test = load_split("test", table)
    predicted = top_class(model, X_test)
    classes = list(model.classes_)
    report_text = classification_report(y_test, predicted, labels=classes, digits=3)
    report_dict = classification_report(y_test, predicted, labels=classes, output_dict=True)
    test_metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "macro_f1": f1_score(y_test, predicted, average="macro"),
        "per_class": {c: report_dict[c] for c in classes},  # precision, recall, F1, support
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=classes).tolist(),
        "n_images": int(len(y_test)),
    }
    print()
    print(report_text)
    print(f"  Accuracy:          {test_metrics['accuracy']:.3f}")
    print(f"  Balanced accuracy: {test_metrics['balanced_accuracy']:.3f}")

    # --- save model, metadata and report ---
    model_file = MODELS_DIR / f"model_{MODEL_VERSION}.joblib"
    if model_file.exists():
        print(f"\n  Note: {model_file} already existed and is replaced.")
    joblib.dump(model, model_file)

    metadata = {
        "version": MODEL_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_type": best_name,
        "classes": classes,
        "preprocessing": preprocessing.settings(),
        "trained_on": "train + validation" if REFIT_ON_TRAIN_AND_VAL else "train",
        "n_training_images": int(len(y_fit)),
        "validation_comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "random_seed": RANDOM_SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    metadata_file = MODELS_DIR / f"model_{MODEL_VERSION}.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    report_file = DOCS_DIR / f"model_report_{MODEL_VERSION}.txt"
    report_file.write_text(
        f"Model {MODEL_VERSION} ({best_name}), created {metadata['created']}\n\n"
        "Validation comparison (trained on train, scored on validation):\n"
        f"{comparison.to_string(index=False, float_format='%.3f')}\n\n"
        f"Test set ({len(y_test)} images, used once):\n{report_text}\n"
        f"Accuracy:          {test_metrics['accuracy']:.3f}\n"
        f"Balanced accuracy: {test_metrics['balanced_accuracy']:.3f}\n")

    matrix_file = DOCS_DIR / f"confusion_matrix_{MODEL_VERSION}.png"
    save_confusion_matrix(y_test, predicted, classes, matrix_file)

    size_mb = model_file.stat().st_size / 1_000_000
    print("\nSaved:")
    print(f"  {model_file}  ({size_mb:.1f} MB)")
    print(f"  {metadata_file}")
    print(f"  {report_file}")
    print(f"  {matrix_file}")


if __name__ == "__main__":
    main()