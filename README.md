# DLBDSMTP01_Project-From-Model-to-Production-Environment
THe repository for my Project from the course DLBDSMTP01

# Refund Item Classification – From Model to Production

Course project for DLBDSMTP01 (Project: From Model to Production), Task 2.

An image classification model sorts returned items into categories. The model runs as a REST service and automatically processes all new images in a nightly batch.

> **Status:** work in progress

## Project structure

| Folder | Content |
|---|---|
| `src/` | Python code: training, API, batch job |
| `models/` | Trained model files (not stored on GitHub) |
| `tests/` | Automated tests |
| `docs/` | Architecture diagram, screenshots, project log |
| `data/` | Image data (not stored on GitHub, see "Data") |

## Setup

Requirements: Python 3.11 or 3.12, Git.

```bash
git clone https://github.com/nosadfountains/DLBDSMTP01_Project-From-Model-to-Production-Environment.git 
cd DLBDSMTP01_Project-From-Model-to-Production-Environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data

## Data

This project uses the **Apparel images dataset** by trolukovich, published on Kaggle under the CC0: Public Domain licence:
https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset

The dataset contains 11,385 images in 24 folders named `colour_item` (e.g. `black_dress`). For this project the colours are merged into five item classes:

| Class | Images |
|---|---|
| shoes | 3,418 |
| pants | 2,789 |
| dress | 2,570 |
| shirt | 1,686 |
| shorts | 922 |

The images are not stored in this repository. To reproduce:

1. Log in to Kaggle and download the dataset as a zip file (about 270 MB) from the link above.
2. Unzip it into `data/raw/` inside the project folder:
   ```bash
   mkdir -p data/raw
   unzip -q path/to/archive.zip -d data/raw
   ```
3. Check the data:
   ```bash
   python src/check_data.py
   ```
   Results you should have: 11,385 images, 0 broken.

## Usage

_To be added: training the model, starting the API, running the batch job._