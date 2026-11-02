"""
Quick check of the raw dataset in data/raw/.

For every class folder it counts the images, lists files that cannot be
opened, and summarises the image sizes. Nothing is changed or deleted.

Run from the project folder:  python src/check_data.py
"""

from collections import Counter
from pathlib import Path

from PIL import Image

# Folder with one subfolder per class, e.g. data/raw/black_dress
RAW_DIR = Path("data/raw")


def check_folder(folder):
    """Return (number of good images, list of broken files, Counter of sizes)."""
    good = 0
    broken = []
    sizes = Counter()
    for file in folder.iterdir():
        if file.name.startswith("."):  # skip hidden files such as .DS_Store
            continue
        try:
            with Image.open(file) as img:
                sizes[img.size] += 1  # img.size is (width, height)
                img.verify()  # raises an error if the file is damaged
            good += 1
        except Exception as error:
            broken.append((file.name, type(error).__name__))
    return good, broken, sizes


def main():
    if not RAW_DIR.exists():
        print(f"Folder not found: {RAW_DIR.resolve()}")
        print("Run this script from the project folder.")
        return

    folders = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    item_totals = Counter()  # images per item type, colours merged
    all_sizes = Counter()
    all_broken = []

    print(f"{'Folder':<15}{'Images':>8}{'Broken':>8}")
    print("-" * 31)
    for folder in folders:
        good, broken, sizes = check_folder(folder)
        item = folder.name.split("_")[-1]  # black_dress -> dress
        item_totals[item] += good
        all_sizes.update(sizes)
        all_broken += [(folder.name, name, err) for name, err in broken]
        print(f"{folder.name:<15}{good:>8}{len(broken):>8}")
    print("-" * 31)
    print(f"{'Total':<15}{sum(item_totals.values()):>8}{len(all_broken):>8}")

    print("\nImages per item type (colours merged):")
    for item, count in item_totals.most_common():
        print(f"  {item:<10}{count:>6}")

    print("\nMost common image sizes (width x height):")
    for (w, h), count in all_sizes.most_common(5):
        print(f"  {w} x {h}: {count} images")
    widths = [w for (w, h) in all_sizes.elements()]
    heights = [h for (w, h) in all_sizes.elements()]
    if widths:
        print(f"  Width ranges from {min(widths)} to {max(widths)} px")
        print(f"  Height ranges from {min(heights)} to {max(heights)} px")

    if all_broken:
        print("\nFiles that could not be opened:")
        for folder, name, err in all_broken:
            print(f"  {folder}/{name}  ({err})")
    else:
        print("\nAll images could be opened.")


if __name__ == "__main__":
    main()