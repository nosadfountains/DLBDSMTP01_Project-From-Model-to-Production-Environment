"""
Tests for the monitoring script in src/monitor.py.

Every test builds a tiny project in a temporary folder (run summary,
API log, predictions, true classes), so your real logs and results are
never touched.

Run from the project folder:  python -m pytest
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import monitor  # noqa: E402

CLASSES = ["dress", "pants", "shirt", "shoes", "shorts"]
NIGHT = datetime(2026, 11, 18, 22, 15)
MORNING_AFTER = datetime(2026, 11, 19, 8, 0)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point monitor.py at temporary folders instead of the real ones."""
    for name, sub in [("LOG_DIR", "logs"), ("RESULTS_DIR", "results"),
                      ("REPORT_DIR", "reports")]:
        monkeypatch.setattr(monitor, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(monitor, "SPLIT_FILE", tmp_path / "split.csv")
    monkeypatch.setattr(monitor, "LABELS_FILE", tmp_path / "labels.csv")
    monkeypatch.setattr(monitor, "MODEL_INFO_FILE", tmp_path / "missing.json")
    monitor.log.handlers.clear()  # log into the temporary folder

    # training data: 20 % of each class
    pd.DataFrame({"path": [f"x{i}" for i in range(50)],
                  "label": [CLASSES[i % 5] for i in range(50)],
                  "split": "train"}).to_csv(tmp_path / "split.csv", index=False)
    return tmp_path


def add_run(project, started=NIGHT, status="ok", found=50, predicted=50,
            failed=0, left=0):
    file = project / "logs" / "run_summary.csv"
    row = pd.DataFrame([{"started": started.isoformat(), "status": status,
                         "images_found": found, "predicted": predicted,
                         "failed_images": failed, "left_in_incoming": left,
                         "model_version": "v1", "duration_s": 0.5,
                         "results_file": f"predictions_{started:%Y-%m-%d}.csv"}])
    row.to_csv(file, mode="a", header=not file.exists(), index=False)


def add_predictions(project, top_classes, confidences, true_classes=None,
                    night=NIGHT):
    """Write a predictions file; optionally the true classes too."""
    rows = []
    for i, (top, conf) in enumerate(zip(top_classes, confidences)):
        rest = (1 - conf) / 4
        rows.append({"image_id": f"img{i}.jpg", "processed_at": night.isoformat(),
                     "model_version": "v1", "top_class": top, "confidence": conf,
                     **{f"p_{c}": conf if c == top else rest for c in CLASSES}})
    pd.DataFrame(rows).to_csv(project / "results" / f"predictions_{night:%Y-%m-%d}.csv",
                              index=False)
    if true_classes is not None:
        pd.DataFrame({"file": [f"img{i}.jpg" for i in range(len(true_classes))],
                      "path": "x", "label": true_classes,
                      "delivered_at": night.isoformat()}).to_csv(project / "labels.csv",
                                                                index=False)


def a_normal_night(project):
    """50 images, every class 20 %, all confident and correct."""
    tops = [CLASSES[i % 5] for i in range(50)]
    add_run(project)
    add_predictions(project, tops, [0.9] * 48 + [0.3, 0.45], true_classes=tops)


def test_normal_night_has_no_alerts_and_writes_everything(project):
    a_normal_night(project)
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert alerts == []
    for name in ["dashboard.png", "monitoring_report.txt", "nightly_metrics.csv"]:
        assert (project / "reports" / name).exists()


def test_review_queue_has_only_unsure_images_most_unsure_first(project):
    a_normal_night(project)
    monitor.run_monitoring(now=MORNING_AFTER)
    queue = pd.read_csv(project / "results" / "review_queue_2026-11-18.csv")
    assert list(queue["confidence"]) == [0.3, 0.45]
    assert "checked_class" in queue.columns  # empty column for the reviewer
    assert (queue["second_class"] != queue["top_class"]).all()


def test_failed_last_run_is_critical(project):
    a_normal_night(project)
    add_run(project, started=datetime(2026, 11, 19, 7, 0), status="api_unavailable",
            predicted=0, left=50)
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert ("CRITICAL" in [level for level, _ in alerts])
    assert any("still waiting" in message for _, message in alerts)


def test_missing_night_is_noticed(project):
    a_normal_night(project)
    alerts = monitor.run_monitoring(now=datetime(2026, 11, 20, 8, 0))  # 34 h later
    assert any("No run with new images" in message for _, message in alerts)


def test_class_shift_is_reported_as_possible_drift(project):
    add_run(project)
    add_predictions(project, ["shorts"] * 40 + ["shoes"] * 10, [0.9] * 50)
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert any("'shorts'" in message and "drift" in message for _, message in alerts)


def test_accuracy_drop_is_reported(project):
    tops = [CLASSES[i % 5] for i in range(50)]
    truth = tops[:25] + [CLASSES[(i + 1) % 5] for i in range(25, 50)]  # half wrong
    add_run(project)
    add_predictions(project, tops, [0.9] * 50, true_classes=truth)
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert any("accuracy 50.0%" in message for _, message in alerts)


def test_many_unsure_images_are_reported(project):
    add_run(project)
    add_predictions(project, [CLASSES[i % 5] for i in range(50)], [0.4] * 50)
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert any("below confidence" in message for _, message in alerts)


def test_threshold_table_counts_caught_errors():
    predictions = pd.DataFrame({"top_class": ["a", "a", "a", "a"],
                                "true_class": ["a", "b", "a", "b"],
                                "confidence": [0.9, 0.3, 0.8, 0.55]})
    table = monitor.threshold_table(predictions).set_index("threshold")
    assert table.loc[0.5, "sent_to_review"] == "25%"   # only the 0.3 image
    assert table.loc[0.5, "errors_caught"] == "50%"    # 1 of the 2 wrong ones
    assert table.loc[0.6, "errors_caught"] == "100%"


def test_empty_project_does_not_crash(project):
    alerts = monitor.run_monitoring(now=MORNING_AFTER)
    assert alerts == [("CRITICAL", "No batch run found in logs/run_summary.csv")]