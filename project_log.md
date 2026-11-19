# Project Log

## Setup: GitHub repository
- **Did:** Created the GitHub repository and cloned it in VS Code
- **Problem:** "Publish to GitHub" button was not shown in VS Code; the Cmd+Shift+P shortcut did not work
- **Solution:** Created the repository on github.com first, then cloned it via View → Command Palette → "Git: Clone"


## Setup: environment and project structure
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


### WP3 – Dataset
- **Did:** Compared three Kaggle datasets (Apparel images, Fashion Product Images (Small), Clothing dataset full); downloaded the chosen one into `data/raw/`; wrote `src/check_data.py` to check it
- **Decision:** Apparel images dataset (trolukovich, CC0); colours merged into five classes: shoes, pants, dress, shirt, shorts
- **Reasons:** simple folder-per-class structure, clear licence, manageable size (about 270 MB), enough images per class; keeps the focus on the API and batch processing, as the task asks
- **Rejected:** Fashion Product Images (Small): many categories, but labels in a CSV file, strongly imbalanced, very small studio images, originally scraped from an online shop. Clothing dataset (full): closest to real refund photos and CC0, but several classes too small and some unclear labels
- **Data check result:** 11,385 images, 0 broken; class imbalance (shoes 3,418 vs shorts 922); image sizes vary (width 117–474 px, height 154–1,377 px)
- **Limitations to mention in the talk:** colour gaps (e.g. no red shirts, no green dresses); web/studio images differ from real refund photos; the origin of the images is not stated on Kaggle
- **Open decisions for WP4:** how to handle the class imbalance; how to bring all images to one size
- **Reference list entry (to format per IU citation guidelines):** trolukovich, "Apparel images dataset", Kaggle, licence CC0: Public Domain, https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset

### Data split
- **Did:** Wrote `src/split_data.py`; it writes `data/split.csv` (one row per image: path, colour, class, split). Images are not copied or moved
- **Decision:** 60 % train, 15 % validation, 15 % test, 10 % "incoming" pool that is never used in training and feeds the simulated nightly batch (WP6); fixed seed 42 so the split can be reproduced
- **Decision:** Split per colour_item folder (stratified), so rare combinations such as brown shorts appear in every part
- **Problem:** The dataset contained 11 extra copies of identical images and one image saved under two different classes
- **Solution:** Removed exact duplicates before splitting (prevents the same image appearing in training and test); dropped both copies of the image with conflicting classes
- **Result:** 11,372 images: train 6,822, validation 1,706, test 1,706, incoming 1,138


## WP2 Conceptual architecture (v1)

Diagram: docs/architecture_v1.png

| Decision | Choice | Reason | Alternatives considered |
|---|---|---|---|
| Trigger | cron nightly + manual "run now" command (same script) | Task requires nightly batch runs; manual run lets the supervisor and the demo see results immediately | launchd (catches missed runs, harder setup); GitHub Actions (cannot reach local images or API); event-driven (not batch) |
| Batch job → model | HTTP calls to the Flask API | Keeps the model-as-a-service separation | Load model file directly (simpler, weaker design) |
| Image storage | data/incoming → data/processed, corrupt files → data/failed | Simple, visible, prevents double processing | Track processed IDs in a file |
| Results | One CSV per night in results/ | Opens in Excel for the refund team | SQLite (better for queries across nights) |
| Model storage | Versioned file in models/ + metadata file | No extra tool | mlflow model registry |

Open point: cron skips runs while the Mac is asleep → handle in WP6.

## WP4 Model (v1)

- **Did:** Wrote `src/preprocessing.py` (shared image preparation for training and the API), `src/train.py` (compares models, trains the final one, evaluates it once on the test set) and `src/evaluate_model.py` (reloads the saved model and checks the test numbers); added `tests/test_preprocessing.py` (4 tests, all passed)
- **Decision:** Classic machine learning with scikit-learn, as planned at the start; the task asks for a simple model checked with basic statistics
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


## WP5 Model as a REST service

- **Did:** Wrote `src/api.py` (Flask service with `GET /health`, `GET /model-info`, `POST /predict`), `src/predict_client.py` (sends images to the API and prints the answer) and `tests/test_api.py` (8 tests: valid batch, API equals direct model prediction, corrupt image, empty request, too many images, no model, wrong address)
- **Decision:** Images are sent as a normal file upload (multipart, field name `images`); the answer is JSON with id, status, top class, confidence and the probability of every class
- **Reason:** simplest option; no extra encoding step and easy to test with curl. Alternative: Base64 inside JSON (one uniform format, but about a third larger and an extra encode/decode step)
- **Decision:** A corrupt image only marks that image as `"status": "error"`; the other images in the request are still predicted
- **Reason:** the batch job (WP6) can move exactly the bad image to `data/failed/`. Alternative: reject the whole request (simpler code, but one bad photo would block a whole chunk)
- **Decision:** At most 64 images (and 50 MB) per request; the batch job sends a night in chunks
- **Decision:** Error answers: 400 no images, 413 too many images, 503 no model loaded, 404 wrong address, 500 unexpected error (full details only in the log, short message to the client)
- **Decision:** Every request is logged in `logs/api.log` (number of images, ok, failed, duration); `logs/` is in `.gitignore`
- **Decision:** The API uses the same `src/preprocessing.py` as training and refuses to load a model whose preprocessing settings differ
- **Decision:** No Docker (scope decision from WP2)
- **Problem avoided:** Port 5000, Flask's default, is used by AirPlay on macOS; the API runs on port 8000 instead
- **Result:** 5 random test images predicted with the real model; a text file sent as an image came back as `ERROR: not a readable image` while the API kept running
- **Observation:** green shorts were predicted as shoes with 51 % confidence and a blue dress as dress with only 46 % → idea for WP7: send low-confidence images to a person for review


## WP6 Nightly batch processing

- **Did:** Wrote `src/batch_predict.py` (sends waiting images to the API in chunks and saves the results), `src/simulate_night.py` (delivers new "refund photos" from the incoming pool), `run_nightly.sh` (starts the API, runs the batch job, stops the API) and `tests/test_batch.py` (5 tests); 17 tests pass in total
- **Decision:** Images wait in `data/incoming/`; predicted images move to `data/processed/`, unreadable ones to `data/failed/`; results go to `results/predictions_<date>.csv`; every run adds one row to `logs/run_summary.csv`
- **Decision:** Results are saved first, then the image is moved, so a crash can at worst repeat a row but never lose a result
- **Decision:** If the API is down or a chunk fails, the remaining images stay in `data/incoming/` for the next run (idempotent: rerunning is always safe)
- **Decision:** The API runs only during the nightly job (started and stopped by `run_nightly.sh`). Alternative: an API that runs all the time (more realistic, but a terminal or service would have to stay open)
- **Decision:** `simulate_night.py` writes the true class of each delivered image to `data/delivered_labels.csv`; the batch job never reads it; it is kept for checking nightly accuracy in WP7
- **Decision (trigger):** cron. Tested with a run every 5 minutes; final schedule: 100 new images at 18:00, batch run at 22:15, after most shops close at 22:00
- **Constraint:** cron only runs while the Mac is on. I cannot keep it on overnight, so runs happen on evenings it is on; a missed evening loses nothing because the images wait for the next run. In production this would run on a server or in the cloud (optional WP8)
- **Results:** manual run: 10 images predicted in 0.2 s, status ok; rerun: 0 new images (nothing predicted twice); cron run at 08:20 without me: 10 images, exit code 0
- **Problem:** Two commands pasted onto one line (`./run_nightly.shcat ...`) → "No such file or directory"
- **Solution:** Run one command per line
- **Problem:** The "prevent sleep" setting was not under Batterie → Optionen on my macOS
- **Solution:** Found it under Displays → Erweitert ("Kein automatisches Aktivieren des Ruhezustands im Netzbetrieb bei ausgeschaltetem Display")
- **Observation:** 3 of the first 10 images had a top-class confidence below 0.55 (0.27, 0.33, 0.52) → review queue in WP7


## WP7 Monitoring

- **Did:** Wrote `src/monitor.py` (reads the logs and results, checks them, writes a report, a dashboard picture and a review queue) and `tests/test_monitor.py` (9 tests: normal night, review queue, failed run, missed night, drift, accuracy drop, many unsure images, threshold table, empty project); all 26 tests passed. `run_nightly.sh` now runs the monitoring after every batch; `reports/` is in `.gitignore`
- **Decision:** Monitoring is a separate script that only reads what the API and the batch job write
- **Reason:** it cannot break the tested batch job, and it can also be run by hand. Alternative: checks inside `batch_predict.py` (one file less, but changes tested code)
- **Decision:** Technical checks: last run failed → critical; no run with new images for 26 h, images left in `data/incoming/`, unreadable images, API error lines, a request slower than 5 s → warning
- **Decision:** Model checks, only for nights with at least 30 images: a class share more than 15 percentage points away from training → possible drift; more than 30 % of images below confidence 0.5 → warning; accuracy more than 10 points below the test set → warning
- **Reason:** with about 100 images a night the numbers move by chance; smaller nights would give false alarms
- **Decision:** Images with a confidence below 0.5 go into `results/review_queue_<date>.csv` for a person to check (most unsure first, with the second most likely class and an empty column for the checked class)
- **Ground-truth gap:** in real life the true class of a nightly photo is unknown. Here `data/delivered_labels.csv` plays the role of staff who check photos by hand; the batch job never reads it
- **Result:** first scheduled night: 120 images, accuracy 85.8 % (test set 85.7 %), no alerts. Three manual "run now" batches: 300 images, accuracy 86.3 %, no alerts
- **Threshold comparison (420 checked images):**

  | Threshold | Sent to review | Mistakes caught | Accuracy of the rest |
  |---|---|---|---|
  | 0.4 | 11 % | 47 % | 91.7 % |
  | 0.5 (chosen) | 21 % | 74 % | 95.5 % |
  | 0.6 | 33 % | 86 % | 97.2 % |
  | 0.7 | 43 % | 93 % | 98.3 % |

- **Decision:** Keep 0.5: from 0.4 to 0.5 every 2.6 extra checks catch one more mistake, from 0.5 to 0.6 it takes about 7. If a wrong category were expensive, 0.6 would be the safer choice. The alert limit stays at 30 % (normal nights: 19–24 %)
- **Not done (by choice):** e-mail alerts (nobody is notified actively; alerts are in the report and `logs/monitor.log`), mlflow (one more tool)
- **Constraints for the talk:** the API has no login and no HTTPS; it only listens on the Mac itself (127.0.0.1). Production would need an API key, HTTPS and access rights on `results/`; refund photos may show personal data. The manual batches are not real nights
- **Problem:** typed two commands on one line (`./run_nightly.shpython ...`) → "No such file or directory"
- **Solution:** one command per line


## Re-planning

- **Decision:** Cloud deployment (Task step 6, optional) is not done. The plan's condition for it was met (WP1–WP7 done, buffer left), but I decided against it
- **Reason:** not needed for the top grade; it would cost several days, possibly money and new sources of errors. In the talk the cloud is named as the next step, because it would remove the "laptop must be on" constraint
- **Consequence:** the work packages after WP7 were renumbered (WP9–WP12 are now WP8–WP11)

## WP8 Documentation and reproducibility

- **Did:** Rewrote `README.md` (purpose, architecture, setup, data, training, API with example requests and error codes, run now and cron, monitoring, tests, results, limitations, sources); drew `docs/architecture_v2.png` (the system as built, including arrivals, monitoring and review queue); renamed `docs/architecture_v1.png.png` to `docs/architecture_v1.png`
- **Fresh-clone test:** the repository was cloned into an empty folder (Linux, Python 3.13), `requirements.txt` installed, all 26 tests passed, every README step ran on stand-in images, and the API started with the uploaded model without training. A second test on my Mac was skipped by choice: the Mac already runs the same pinned versions every day
- **Problem found:** the README said "Python 3.11 or 3.12", but the pinned numpy and scipy versions need Python 3.12 or newer
- **Solution:** the README now says "3.12 or newer (tested with 3.14 on macOS and 3.13 on Linux)"
- **Decision:** the trained model (`models/model_v1.joblib`, 14.9 MB, and `model_v1.json`) is on GitHub; `.gitignore` lets only version 1 through
- **Reason:** the API can be started right after setup, with exactly the graded model. Against it: the file stays in the Git history and only loads reliably with the pinned scikit-learn version. Alternative: keep it local and let the reader train it (a few minutes)
- **Problem:** `git push` of the model failed with "RPC failed; HTTP 400"
- **Solution:** `git config http.postBuffer 524288000`; Git's default buffer of 1 MB is too small for the 15 MB file
- **Problem:** the new README and the diagram were not on GitHub after the push, because the downloaded files had not reached the project folder; the commit was "clean"
- **Solution:** checked with `git status` and `ls docs` before each push; the files were put into the project folder directly
- **Check:** the repository is public (it downloads without a login)
- **Evidence:** `docs/evidence/` holds real outputs for the slides: API answers (health, model info, a request with two photos, a 400 error, a fake photo marked as error while the real one is still predicted), the crontab, one scheduled evening (18:00 and 22:15), the run summary, excerpts of results and review queue, and the dashboard; `docs/evidence/README.md` says which slide uses which file
- **Problem:** the evidence commands were pasted all at once and none of them ran
- **Solution:** ran them again one at a time and checked each file
-
## Reflection

### What worked
- A simple model kept the focus on production: the random forest trains in about 5 seconds, the model file is 15 MB and easy to serve
- One shared preprocessing file for training and API, and an API that refuses a model with different settings, prevent silently wrong predictions
- The batch job is safe to rerun: handled photos leave `data/incoming/`, results are saved before a photo is moved; a rerun found 0 new photos
- The model behaves on the incoming photos as on the test set: 85.0 % on 520 checked photos against 85.7 % on the test set, no alerts
- 26 automated tests that need no data, so anyone can check the code in seconds

### Challenges
- Git and GitHub: at the start the repository sat in a subfolder and I worked in an unconnected folder; later commands ran in the wrong folder, two commands ended up on one line, downloaded files never reached the project folder, and a push failed with HTTP 400
- Data: duplicate images and one image with two classes had to be removed; the classes are imbalanced, and shorts stay the weakest class (about 1 in 3 missed)
- macOS details: port 5000 is used by AirPlay, the sleep setting was in an unexpected place, cron runs only while the Mac is awake

### Constraints
- Everything runs locally on one Mac; runs happen only on evenings it is on
- The API has no login and no HTTPS; it only listens on the Mac itself
- The true classes of nightly photos are simulated; in reality staff would have to check a sample
- Studio and web images, not real refund photos; only one evening ran fully on schedule, the other runs were started by hand

### What I would change
- Run the service on a server or in the cloud, so it runs every night; on macOS, launchd instead of cron would catch missed runs
- Turn the review queue into a feedback loop: photos checked by staff become new labelled data for the next model version, with a model registry such as mlflow
- Try a pretrained neural network once the pipeline is stable, for higher accuracy, especially for shorts
- Results in a small database (SQLite) instead of one CSV per night, to compare nights easily
- Active alerts, e.g. an e-mail when a run fails
- From day one: check `pwd` and `git status` before every command that changes files

### Personal learning
- What really surprised me are the little things to keep a model running. Due to the contents of my studies, I had plenty of code and techniques at hand to create the model and the evaluation, but the connects, the architecture, the interplay between the running model, the test runs, the data, git hub, all provided room for errors and often took me longer to fix than the actual model code.

