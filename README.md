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

_To be added: dataset source, licence and download instructions._

## Usage

_To be added: training the model, starting the API, running the batch job._