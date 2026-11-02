# Project Log

## 2026-10-29
- **Did:** Created the GitHub repository and cloned it in VS Code
- **Problem:** "Publish to GitHub" button was not shown in VS Code; the Cmd+Shift+P shortcut did not work
- **Solution:** Created the repository on github.com first, then cloned it via View → Command Palette → "Git: Clone"
- **Hours:**

## 2026-11-02
- **Did:** Set up virtual environment, base packages, requirements.txt, folder structure, .gitignore, README and project log
- **Problem:** `python: command not found`
- **Solution:** The environment was not active; activated it with `source .venv/bin/activate`, later set it as the VS Code interpreter so it activates automatically
- **Problem:** Accidentally ran commands in a different project folder
- **Solution:** Checked the location with `pwd` before running commands
- **Problem:** A failed `pip freeze > requirements.txt` left an empty file
- **Learned:** `>` empties the target file before the command runs
- **Problem:** Sync to GitHub stopped working; `git status` reported "not a git repository"
- **Cause:** "Git: Clone" had created the repository as a subfolder; all work was being saved in the parent folder, which is not connected to GitHub
- **Solution:** Moved the repository to `~/Projects` (a path without spaces), moved the work files into it, recreated the virtual environment there
- **Problem:** The repository had no .gitignore, so the virtual environment could have been uploaded
- **Solution:** Added a .gitignore for `.venv/`, `data/`, model files, cache files, secrets and `.DS_Store`
- **Problem:** Test file name contained `#` and spaces, which breaks Python imports and terminal commands
- **Solution:** Deleted the file
- **Hours: 1:30h

### WP3 – Dataset (2026-11-02)
- **Did:** Compared three Kaggle datasets (Apparel images, Fashion Product Images (Small), Clothing dataset full); downloaded the chosen one into `data/raw/`; wrote `src/check_data.py` to check it
- **Decision:** Apparel images dataset (trolukovich, CC0); colours merged into five classes: shoes, pants, dress, shirt, shorts
- **Reasons:** simple folder-per-class structure, clear licence, manageable size (about 270 MB), enough images per class; keeps the focus on the API and batch processing, as the task asks
- **Rejected:** Fashion Product Images (Small): many categories, but labels in a CSV file, strongly imbalanced, very small studio images, originally scraped from an online shop. Clothing dataset (full): closest to real refund photos and CC0, but several classes too small and some unclear labels
- **Data check result:** 11,385 images, 0 broken; class imbalance (shoes 3,418 vs shorts 922); image sizes vary (width 117–474 px, height 154–1,377 px)
- **Limitations to mention in the talk:** colour gaps (e.g. no red shirts, no green dresses); web/studio images differ from real refund photos; the origin of the images is not stated on Kaggle
- **Open decisions for WP4:** how to handle the class imbalance; how to bring all images to one size
- **Reference list entry (to format per IU citation guidelines):** trolukovich, "Apparel images dataset", Kaggle, licence CC0: Public Domain, https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset, accessed 2026-11-02