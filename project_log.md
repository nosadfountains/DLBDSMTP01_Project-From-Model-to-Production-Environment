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

### Data split (2026-11-16)
- **Did:** Wrote `src/split_data.py`; it writes `data/split.csv` (one row per image: path, colour, class, split). Images are not copied or moved
- **Decision:** 60 % train, 15 % validation, 15 % test, 10 % "incoming" pool that is never used in training and feeds the simulated nightly batch (WP6); fixed seed 42 so the split can be reproduced
- **Decision:** Split per colour_item folder (stratified), so rare combinations such as brown shorts appear in every part
- **Problem:** The dataset contained 11 extra copies of identical images and one image saved under two different classes
- **Solution:** Removed exact duplicates before splitting (prevents the same image appearing in training and test); dropped both copies of the image with conflicting classes
- **Result:** 11,372 images: train 6,822, validation 1,706, test 1,706, incoming 1,138
- **Hours:2**

## 2026-11-16 – WP2 Conceptual architecture (v1)

Diagram: docs/architecture_v1.png

| Decision | Choice | Reason | Alternatives considered |
|---|---|---|---|
| Trigger | cron nightly + manual "run now" command (same script) | Task requires nightly batch runs; manual run lets the supervisor and the demo see results immediately | launchd (catches missed runs, harder setup); GitHub Actions (cannot reach local images or API); event-driven (not batch) |
| Batch job → model | HTTP calls to the Flask API | Keeps the model-as-a-service separation | Load model file directly (simpler, weaker design) |
| Image storage | data/incoming → data/processed, corrupt files → data/failed | Simple, visible, prevents double processing | Track processed IDs in a file |
| Results | One CSV per night in results/ | Opens in Excel for the refund team | SQLite (better for queries across nights) |
| Model storage | Versioned file in models/ + metadata file | No extra tool | mlflow model registry |

Open point: cron skips runs while the Mac is asleep → handle in WP6.

## 2026-11-17 – WP4 Model (v1)

- **Did:** Wrote `src/preprocessing.py` (shared image preparation for training and the API), `src/train.py` (compares models, trains the final one, evaluates it once on the test set) and `src/evaluate_model.py` (reloads the saved model and checks the test numbers); added `tests/test_preprocessing.py` (4 tests, all passed)
- **Decision:** Classic machine learning with scikit-learn, as planned on 2 Nov; the task asks for a simple model checked with basic statistics
- **Decision:** Preprocessing: 64×64 px, grayscale, padded to a square with white, pixel values scaled to 0–1. Grayscale because colour does not define the class and the dataset has colour gaps; padding instead of stretching or cropping so shapes stay intact
- **Decision:** Class imbalance handled with `class_weight="balanced"`; the best model is picked by macro F1, and metrics are reported per class
- **Model comparison (trained on train, scored on validation):**

  | Model | Accuracy | Macro F1 | Training time |
  |---|---|---|---|
  | Random forest (chosen) | 0.847 | 0.825 | 5 s |
  | PCA + SVM | 0.830 | 0.798 | 13 s |
  | PCA + logistic regression | 0.744 | 0.716 | 2 s |
  | Dummy baseline (always the biggest class) | 0.300 | 0.092 | 0 s |

- **Final model:** random forest retrained on train + validation (8,528 images), evaluated once on the test set (1,706 images): accuracy 0.857, balanced accuracy 0.830, macro F1 0.835
- **Per class (test):** shirt F1 0.901, shoes 0.876, pants 0.863, dress 0.858, shorts 0.674. Shorts, the smallest class, is the weakest: recall 0.638, so about 1 in 3 shorts is missed
- **Check:** `src/evaluate_model.py` reloaded the model in a new Python session and reproduced accuracy and macro F1 exactly
- **Saved:** `models/model_v1.joblib` (14.9 MB) and `models/model_v1.json` (classes, preprocessing settings, metrics, versions) stay local; `docs/model_report_v1.txt` and `docs/confusion_matrix_v1.png` are on GitHub
- **Problem:** The first push of `src/preprocessing.py` contained the code of the test file
- **Solution:** Replaced the content and pushed again (commit "error correction"); learned to check the first lines of a file with `head -2` before committing
- **Problem:** scikit-learn 1.9 deprecates `SVC(probability=True)`
- **Solution:** Used `CalibratedClassifierCV(SVC(...), ensemble=False)` instead
- **Limitations for the talk:** lower accuracy than a pretrained neural network would reach; shorts often missed; studio images differ from real refund photos
- **Hours:18

## 2026-11-17 – WP5 Model as a REST service

- **Did:** Wrote `src/api.py` (Flask service with `GET /health`, `GET /model-info`, `POST /predict`), `src/predict_client.py` (sends images to the API and prints the answer) and `tests/test_api.py` (8 tests: valid batch, API equals direct model prediction, corrupt image, empty request, too many images, no model, wrong address)
- **Decision:** Images are sent as a normal file upload (multipart, field name `images`); the answer is JSON with id, status, top class, confidence and the probability of every class
- **Reason:** simplest option; no extra encoding step and easy to test with curl. Alternative: Base64 inside JSON (one uniform format, but about a third larger and an extra encode/decode step)
- **Decision:** A corrupt image only marks that image as `"status": "error"`; the other images in the request are still predicted
- **Reason:** the batch job (WP6) can move exactly the bad image to `data/failed/`. Alternative: reject the whole request (simpler code, but one bad photo would block a whole chunk)
- **Decision:** At most 64 images (and 50 MB) per request; the batch job sends a night in chunks
- **Decision:** Error answers: 400 no images, 413 too many images, 503 no model loaded, 404 wrong address, 500 unexpected error (full details only in the log, short message to the client)
- **Decision:** Every request is logged in `logs/api.log` (number of images, ok, failed, duration); `logs/` is in `.gitignore`
- **Decision:** The API uses the same `src/preprocessing.py` as training and refuses to load a model whose preprocessing settings differ
- **Decision:** No Docker (scope decision of 16 Nov)
- **Problem avoided:** Port 5000, Flask's default, is used by AirPlay on macOS; the API runs on port 8000 instead
- **Result:** 5 random test images predicted with the real model; a text file sent as an image came back as `ERROR: not a readable image` while the API kept running
- **Observation:** green shorts were predicted as shoes with 51 % confidence and a blue dress as dress with only 46 % → idea for WP7: send low-confidence images to a person for review
- **Hours: 6**
