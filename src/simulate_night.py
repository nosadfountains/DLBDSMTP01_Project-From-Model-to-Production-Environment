"""
Simulate the refund photos that arrive during one day (WP6).

Copies a number of images from the "incoming" pool (see data/split.csv)
into data/incoming/, where the nightly batch job picks them up.
Images that were already delivered on an earlier day are skipped, so
every run delivers new photos until the pool of 1,138 images is used up.

The true class of every delivered image is noted in
data/delivered_labels.csv. The batch job never reads this file; it is only
used later to check how good the nightly predictions were (WP7).

Run from the project folder:
  python src/simulate_night.py          100 new images (default)
  python src/simulate_night.py 40       40 new images
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SPLIT_FILE = Path("data/split.csv")
INCOMING_DIR = Path("data/incoming")
LABELS_FILE = Path("data/delivered_labels.csv")
DEFAULT_COUNT = 100
RANDOM_SEED = 42


def main():
    if not SPLIT_FILE.exists():
        print(f"File not found: {SPLIT_FILE.resolve()}")
        print("Run this script from the project folder.")
        return
    count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT

    pool = pd.read_csv(SPLIT_FILE)
    pool = pool[pool["split"] == "incoming"]
    pool = pool.assign(file=pool["path"].map(lambda p: Path(p).name))

    delivered = pd.read_csv(LABELS_FILE) if LABELS_FILE.exists() else pd.DataFrame(
        columns=["file", "path", "label", "delivered_at"])
    remaining = pool[~pool["path"].isin(delivered["path"])]
    if remaining.empty:
        print("The incoming pool is used up: all images were delivered before.")
        return

    today = remaining.sample(min(count, len(remaining)), random_state=RANDOM_SEED + len(delivered))
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    for path, file in zip(today["path"], today["file"]):
        shutil.copy2(path, INCOMING_DIR / file)  # copy: data/raw stays complete

    new_rows = pd.DataFrame({"file": today["file"], "path": today["path"],
                             "label": today["label"], "delivered_at": now})
    pd.concat([delivered, new_rows]).to_csv(LABELS_FILE, index=False)

    print(f"Delivered {len(today)} new images to {INCOMING_DIR}/")
    print(f"Still in the pool: {len(remaining) - len(today)} of {len(pool)}")


if __name__ == "__main__":
    main()