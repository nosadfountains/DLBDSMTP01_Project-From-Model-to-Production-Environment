"""
Split the raw dataset into train, validation, test and an "incoming" pool.

The images are NOT copied or moved. The script writes one table,
data/split.csv, with one row per image: its path, colour, class and split.

  train     - used to train the model (WP4)
  val       - used to compare model settings during training (WP4)
  test      - used once at the end to report the final accuracy (WP4)
  incoming  - never used in training; plays the role of new refund
              photos that arrive every night (WP6 nightly batch)

Exact duplicate images are removed first, so the same picture cannot end
up in both the training data and the test data.

Run from the project folder:  python src/split_data.py
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/split.csv")

INCOMING_SHARE = 0.10  # 10 % of all images go to the incoming pool
VAL_SHARE = 0.15       # 15 % of all images for validation
TEST_SHARE = 0.15      # 15 % of all images for testing; the rest is training
RANDOM_SEED = 42       # fixed seed: the same split every time the script runs


def file_hash(path):
    """Return a fingerprint of the file content (same content = same hash)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_images():
    """Return a table with one row per image file in data/raw/."""
    rows = []
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        colour, item = folder.name.rsplit("_", 1)  # black_dress -> black, dress
        for file in sorted(folder.iterdir()):
            if file.name.startswith("."):  # skip hidden files such as .DS_Store
                continue
            rows.append({
                "path": file.as_posix(),
                "folder": folder.name,
                "colour": colour,
                "label": item,
                "hash": file_hash(file),
            })
    return pd.DataFrame(rows)


def remove_duplicates(images):
    """Keep one copy of identical images; drop images whose copies disagree on the class."""
    labels_per_hash = images.groupby("hash")["label"].nunique()
    conflicting = labels_per_hash[labels_per_hash > 1].index
    copies = images.duplicated("hash").sum()

    cleaned = images[~images["hash"].isin(conflicting)]
    cleaned = cleaned.drop_duplicates("hash", keep="first")

    print(f"Images found:                       {len(images):>6}")
    print(f"Duplicate copies found:             {copies:>6}")
    print(f"Images with conflicting classes:    {images['hash'].isin(conflicting).sum():>6}")
    print(f"Images left after cleaning:         {len(cleaned):>6}\n")
    return cleaned


def split(images):
    """Add a 'split' column. Each split keeps the same mix of colour/item folders."""
    # Step 1: set aside the incoming pool
    rest, incoming = train_test_split(
        images, test_size=INCOMING_SHARE,
        stratify=images["folder"], random_state=RANDOM_SEED)

    # Step 2: set aside the test set (share is converted to "share of what is left")
    rest, test = train_test_split(
        rest, test_size=TEST_SHARE / (1 - INCOMING_SHARE),
        stratify=rest["folder"], random_state=RANDOM_SEED)

    # Step 3: split the remainder into train and validation
    train, val = train_test_split(
        rest, test_size=VAL_SHARE / (1 - INCOMING_SHARE - TEST_SHARE),
        stratify=rest["folder"], random_state=RANDOM_SEED)

    parts = [train.assign(split="train"), val.assign(split="val"),
             test.assign(split="test"), incoming.assign(split="incoming")]
    return pd.concat(parts).sort_values("path")


def main():
    if not RAW_DIR.exists():
        print(f"Folder not found: {RAW_DIR.resolve()}")
        print("Run this script from the project folder.")
        return

    images = remove_duplicates(list_images())
    result = split(images)
    result[["path", "colour", "label", "split"]].to_csv(OUTPUT_FILE, index=False)

    order = ["train", "val", "test", "incoming"]
    table = pd.crosstab(result["label"], result["split"], margins=True,
                        margins_name="total")[order + ["total"]]
    print("Images per class and split:")
    print(table.to_string())
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()