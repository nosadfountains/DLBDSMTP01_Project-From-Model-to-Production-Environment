"""
Monitoring of the nightly batch service (WP7).

Answers one question every morning: did last night's run work, and can we
still trust the model? It only READS what the API and the batch job already
write, so it cannot break them:

  logs/run_summary.csv          one row per batch run       (technical health)
  logs/api.log                  one line per API request    (technical health)
  results/predictions_<date>.csv  the predictions           (model behaviour)
  data/split.csv                class mix the model learned (baseline for drift)
  data/delivered_labels.csv     true classes of the simulated arrivals
                                (stands in for staff spot checks, see below)
  models/model_<version>.json   test accuracy of the model  (baseline)

It writes:

  results/review_queue_<date>.csv  images the model was unsure about, for a
                                   person to check (one file per night)
  reports/nightly_metrics.csv      one row per night with all numbers
  reports/monitoring_report.txt    alerts + tables (also printed)
  reports/dashboard.png            one picture for the slides
  logs/monitor.log                 one line per alert

Why delivered_labels.csv: in real life nobody knows the true class of a
nightly photo, so accuracy cannot be measured directly. Here the simulated
arrivals come with their true class, which plays the role of staff who
check a sample of photos by hand. The batch job never reads this file.

Run from any folder:      python src/monitor.py
run_nightly.sh also runs it after every batch.
"""

import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw into a file; cron has no screen
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent  # works from any folder (cron)
LOG_DIR = PROJECT_DIR / "logs"
RESULTS_DIR = PROJECT_DIR / "results"
REPORT_DIR = PROJECT_DIR / "reports"
SPLIT_FILE = PROJECT_DIR / "data" / "split.csv"
LABELS_FILE = PROJECT_DIR / "data" / "delivered_labels.csv"
MODEL_INFO_FILE = PROJECT_DIR / "models" / "model_v1.json"

# --- Thresholds: the decisions of WP7, all in one place --------------------
REVIEW_THRESHOLD = 0.50     # confidence below this -> a person checks the image
MAX_HOURS_SINCE_RUN = 26    # no run with images for longer -> alert (night + margin)
MAX_REQUEST_MS = 5000       # one API request slower than this -> alert
MAX_CLASS_SHIFT = 0.15      # a class share 15 percentage points off training -> alert
MAX_REVIEW_SHARE = 0.30     # more than 30 % of a night sent to review -> alert
MAX_ACCURACY_DROP = 0.10    # accuracy 10 points below the test set -> alert
MIN_IMAGES_FOR_MODEL_CHECKS = 30  # smaller nights are too noisy to judge
DEFAULT_TEST_ACCURACY = 0.857     # used if the model's .json file is missing
THRESHOLDS_TO_COMPARE = [0.40, 0.50, 0.60, 0.70]

log = logging.getLogger("monitor")


def setup_logging():
    if log.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "monitor.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# --- Reading what the system wrote ------------------------------------------

def read_runs():
    """logs/run_summary.csv as a table; empty table if there is none yet."""
    file = LOG_DIR / "run_summary.csv"
    if not file.exists():
        return pd.DataFrame()
    runs = pd.read_csv(file)
    runs["started"] = pd.to_datetime(runs["started"])
    runs["night"] = runs["started"].dt.date
    return runs


def read_api_log():
    """One row per line of logs/api.log: time, level, request numbers if any."""
    file = LOG_DIR / "api.log"
    if not file.exists():
        return pd.DataFrame(columns=["time", "level", "images", "failed", "duration_ms"])
    pattern = re.compile(r"^(\S+ \S+?),\d+ (\w+) (.*)$")
    rows = []
    for line in file.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue  # e.g. lines of a traceback
        time, level, message = match.groups()
        numbers = dict(re.findall(r"(\w+)=([\d.]+)", message)) if message.startswith("predict ") else {}
        rows.append({"time": pd.to_datetime(time, errors="coerce"), "level": level,
                     "images": float(numbers.get("images", "nan")),
                     "failed": float(numbers.get("failed", "nan")),
                     "duration_ms": float(numbers.get("duration_ms", "nan"))})
    return pd.DataFrame(rows).dropna(subset=["time"])


def read_predictions():
    """All results/predictions_<date>.csv files in one table."""
    frames = []
    for file in sorted(RESULTS_DIR.glob("predictions_*.csv")):
        frame = pd.read_csv(file)
        frame["night"] = pd.to_datetime(file.stem.removeprefix("predictions_")).date()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def training_class_shares():
    """Share of each class in the training data (what the model expects)."""
    if not SPLIT_FILE.exists():
        return None
    split = pd.read_csv(SPLIT_FILE)
    return split.loc[split["split"] == "train", "label"].value_counts(normalize=True)


def test_accuracy():
    try:
        return json.loads(MODEL_INFO_FILE.read_text())["test_metrics"]["accuracy"]
    except Exception:
        return DEFAULT_TEST_ACCURACY


def add_true_labels(predictions):
    """Add the column 'true_class' where the true class is known."""
    if predictions.empty or not LABELS_FILE.exists():
        return predictions.assign(true_class=pd.NA)
    labels = (pd.read_csv(LABELS_FILE).drop_duplicates("file", keep="last")
              .set_index("file")["label"])
    return predictions.assign(true_class=predictions["image_id"].map(labels))


# --- Turning it into numbers --------------------------------------------------

def nightly_metrics(runs, predictions, api_log):
    """One row per night: technical health and model behaviour."""
    nights = set(runs["night"]) if not runs.empty else set()
    nights |= set(predictions["night"]) if not predictions.empty else set()
    rows = []
    for night in sorted(nights):
        row = {"night": night}
        night_runs = runs[runs["night"] == night] if not runs.empty else runs
        if not night_runs.empty:
            row.update({
                "runs": len(night_runs),
                "runs_not_ok": int((night_runs["status"] != "ok").sum()),
                "status": night_runs["status"].iloc[-1],  # how the night ended
                "failed_images": int(night_runs["failed_images"].sum()),
                "left_in_incoming": int(night_runs["left_in_incoming"].iloc[-1]),
                "batch_seconds": float(night_runs["duration_s"].sum()),
            })
        if not api_log.empty:
            day = api_log[api_log["time"].dt.date == night]
            row["api_errors"] = int((day["level"] == "ERROR").sum())
            row["slowest_request_ms"] = day["duration_ms"].max()
        preds = predictions[predictions["night"] == night] if not predictions.empty else predictions
        row["predicted"] = len(preds)
        if len(preds):
            row["mean_confidence"] = round(preds["confidence"].mean(), 3)
            row["review_share"] = round((preds["confidence"] < REVIEW_THRESHOLD).mean(), 3)
            for cls, share in preds["top_class"].value_counts(normalize=True).items():
                row[f"share_{cls}"] = round(share, 3)
            known = preds.dropna(subset=["true_class"])
            row["labelled"] = len(known)
            if len(known):
                row["accuracy"] = round((known["top_class"] == known["true_class"]).mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_table(predictions):
    """For several review thresholds: how many images a person checks,
    how many wrong predictions they catch, how good the rest is."""
    known = predictions.dropna(subset=["true_class"]) if not predictions.empty else predictions
    if known.empty:
        return pd.DataFrame()
    wrong = known["top_class"] != known["true_class"]
    rows = []
    for threshold in THRESHOLDS_TO_COMPARE:
        review = known["confidence"] < threshold
        rows.append({
            "threshold": threshold,
            "sent_to_review": f"{review.mean():.0%}",
            "errors_caught": f"{(wrong & review).sum() / max(wrong.sum(), 1):.0%}",
            "accuracy_of_the_rest": f"{1 - wrong[~review].mean():.1%}" if (~review).any() else "-",
        })
    return pd.DataFrame(rows)


def write_review_queues(predictions):
    """results/review_queue_<date>.csv: unsure images, most unsure first.
    The empty column 'checked_class' is for the person who reviews them."""
    written = []
    if predictions.empty:
        return written
    prob_cols = [c for c in predictions.columns if c.startswith("p_")]
    for night, preds in predictions.groupby("night"):
        unsure = preds[preds["confidence"] < REVIEW_THRESHOLD].copy()
        if prob_cols and len(unsure):
            # the runner-up: the most likely class apart from the top class
            def runner_up(row):
                others = row[prob_cols].drop(f"p_{row['top_class']}", errors="ignore")
                return pd.Series([others.idxmax().removeprefix("p_"), others.max()])
            unsure[["second_class", "second_prob"]] = unsure.apply(runner_up, axis=1)
        queue = unsure.sort_values("confidence")
        columns = [c for c in ["image_id", "top_class", "confidence", "second_class",
                               "second_prob", "processed_at"] if c in queue.columns]
        queue = queue[columns].assign(checked_class="")
        file = RESULTS_DIR / f"review_queue_{night}.csv"
        queue.to_csv(file, index=False)
        written.append((file.name, len(queue)))
    return written


# --- Alerts -------------------------------------------------------------------

def check_alerts(runs, metrics, api_log, now, train_shares, test_acc):
    """Return a list of (level, message). An empty list means all is well."""
    alerts = []
    if runs.empty:
        return [("CRITICAL", "No batch run found in logs/run_summary.csv")]

    # Technical health
    last = runs.iloc[-1]
    if last["status"] != "ok":
        alerts.append(("CRITICAL", f"Last run ({last['started']:%d %b %H:%M}) "
                                   f"ended with status '{last['status']}'"))
    real_runs = runs[runs["images_found"] > 0]
    last_real = real_runs["started"].max() if not real_runs.empty else None
    if last_real is None or now - last_real > timedelta(hours=MAX_HOURS_SINCE_RUN):
        when = "never" if last_real is None else f"{last_real:%d %b %H:%M}"
        alerts.append(("WARNING", f"No run with new images for more than "
                                  f"{MAX_HOURS_SINCE_RUN} h (last: {when})"))
    if last["left_in_incoming"] > 0:
        alerts.append(("WARNING", f"{last['left_in_incoming']} images still waiting "
                                  f"in data/incoming/ after the last run"))
    recent_api = api_log[api_log["time"] > now - timedelta(hours=24)] if not api_log.empty else api_log
    if not recent_api.empty:
        errors = int((recent_api["level"] == "ERROR").sum())
        if errors:
            alerts.append(("WARNING", f"{errors} error line(s) in logs/api.log in the last 24 h"))
        slowest = recent_api["duration_ms"].max()
        if slowest > MAX_REQUEST_MS:
            alerts.append(("WARNING", f"Slowest API request in the last 24 h took {slowest:.0f} ms"))

    # The last night with predictions: failed images and model behaviour
    nights = metrics[metrics["predicted"] > 0] if not metrics.empty else metrics
    if nights.empty:
        return alerts
    night = nights.iloc[-1]
    label = f"{night['night']:%d %b}"
    if night.get("failed_images", 0) > 0:
        alerts.append(("WARNING", f"{label}: {int(night['failed_images'])} unreadable "
                                  f"image(s) moved to data/failed/"))
    if night["predicted"] < MIN_IMAGES_FOR_MODEL_CHECKS:
        return alerts
    if night["review_share"] > MAX_REVIEW_SHARE:
        alerts.append(("WARNING", f"{label}: {night['review_share']:.0%} of images below "
                                  f"confidence {REVIEW_THRESHOLD} (limit {MAX_REVIEW_SHARE:.0%})"))
    if train_shares is not None:
        for cls, expected in train_shares.items():
            seen = night.get(f"share_{cls}", 0)
            seen = 0 if pd.isna(seen) else seen
            if abs(seen - expected) > MAX_CLASS_SHIFT:
                alerts.append(("WARNING", f"{label}: '{cls}' is {seen:.0%} of predictions, "
                                          f"training had {expected:.0%} (possible drift)"))
    if night.get("labelled", 0) >= MIN_IMAGES_FOR_MODEL_CHECKS and \
            night["accuracy"] < test_acc - MAX_ACCURACY_DROP:
        alerts.append(("WARNING", f"{label}: accuracy {night['accuracy']:.1%} on checked "
                                  f"images, test set had {test_acc:.1%}"))
    return alerts


# --- The picture for the slides -----------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CLASS_COLOURS = {"shoes": "#2a78d6", "pants": "#eb6834", "dress": "#1baf7a",
                 "shirt": "#eda100", "shorts": "#e87ba4"}  # fixed order, never by rank
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ref_line(ax, y, text):
    """A dashed reference line, labelled just outside the right edge."""
    ax.axhline(y, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(text, xy=(1.01, y), xycoords=("axes fraction", "data"),
                fontsize=8, color=MUTED, va="center")


def draw_dashboard(metrics, alerts, train_shares, test_acc, file):
    """One picture: headline + four charts over the last 14 calendar days.
    Every day gets a slot, so a night without a run shows up as a gap."""
    done = metrics[metrics["predicted"] > 0]
    if done.empty:
        return False
    days = pd.date_range(end=max(metrics["night"]), periods=14).date
    days = [d for d in days if d >= min(metrics["night"])]
    nights = metrics.set_index("night").reindex(days)
    x = list(range(len(days)))
    labels = [f"{d:%d %b}" for d in days]
    has_run = nights["predicted"].fillna(0) > 0

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)  # 16:9, fits a slide
    fig.patch.set_facecolor("#fcfcfb")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.55, 1, 1], hspace=0.6, wspace=0.22,
                            left=0.05, right=0.88, top=0.93, bottom=0.07)

    # Headline: status of the last night + alerts (icon and word, not colour alone)
    head = fig.add_subplot(grid[0, :])
    head.axis("off")
    last = done.iloc[-1].fillna(0)
    worst = "CRITICAL" if any(a[0] == "CRITICAL" for a in alerts) else \
        "WARNING" if alerts else "OK"
    colour = {"OK": GOOD, "WARNING": WARNING, "CRITICAL": CRITICAL}[worst]
    icon = {"OK": "✓", "WARNING": "!", "CRITICAL": "✕"}[worst]
    head.text(0, 0.95, f"{icon}  {worst}", fontsize=18, weight="bold", color=colour,
              va="top", transform=head.transAxes)
    head.text(0.16, 0.95, f"Last night {last['night']:%a %d %b}: {int(last['predicted'])} "
              f"images predicted, {int(last.get('failed_images', 0))} unreadable, "
              f"{last['review_share']:.0%} sent to review, status '{last.get('status', '?')}'",
              fontsize=11, color=INK, va="top", transform=head.transAxes)
    lines = [f"{a[0].title()}: {a[1]}" for a in alerts[:3]] or ["No alerts"]
    if len(alerts) > 3:
        lines.append(f"... and {len(alerts) - 3} more in reports/monitoring_report.txt")
    head.text(0.16, 0.55, "\n".join(lines), fontsize=9, color="#52514e", va="top",
              transform=head.transAxes, linespacing=1.5)

    def no_run_marks(ax):
        ax.set_xlim(-0.5 if ax.get_xlim()[0] > -1 else -2.2, max(len(x), 7) - 0.5)
        for i in x:
            if not has_run.iloc[i]:
                ax.text(i, 0, "no run", rotation=90, fontsize=7, color=MUTED,
                        ha="center", va="bottom", transform=ax.get_xaxis_transform())

    # 1 Technical health: images per night, problems written above the bar
    ax = fig.add_subplot(grid[1, 0])
    style(ax, "Images predicted per night")
    predicted = nights["predicted"].fillna(0)
    ax.bar(x, predicted, color="#2a78d6", width=0.6)
    top = max(predicted.max(), 1)
    for i, (_, row) in enumerate(nights.iterrows()):
        notes = []
        if row.get("runs_not_ok", 0) > 0:
            notes.append(f"{int(row['runs_not_ok'])} failed run")
        if row.get("failed_images", 0) > 0:
            notes.append(f"{int(row['failed_images'])} unreadable")
        if notes:
            ax.text(i, predicted.iloc[i] + top * 0.03, "! " + "\n".join(notes), fontsize=7,
                    color="#52514e", ha="center", va="bottom")
    ax.set_ylim(0, top * 1.35)
    no_run_marks(ax)
    ax.set_xticks(x, labels)

    # 2 Model behaviour: class mix per night vs. training (parts of a whole)
    ax = fig.add_subplot(grid[1, 1])
    style(ax, "Predicted classes per night (share) vs. training")
    columns = ["Training"] + labels
    positions = [-1.5] + x  # a gap after the training bar
    bottom = [0.0] * len(columns)
    for cls, col in CLASS_COLOURS.items():
        train = train_shares.get(cls, 0) if train_shares is not None else 0
        share = nights[f"share_{cls}"] if f"share_{cls}" in nights else pd.Series(0, index=nights.index)
        values = [train] + list(share.where(has_run, 0).fillna(0))
        ax.bar(positions, values, bottom=bottom, color=col, width=0.6,
               label=cls, edgecolor="#fcfcfb", linewidth=1)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(positions, columns)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    no_run_marks(ax)

    # 3 Share sent to review
    ax = fig.add_subplot(grid[2, 0])
    style(ax, f"Share sent to review (confidence < {REVIEW_THRESHOLD})")
    review = nights["review_share"].fillna(0)
    ax.bar(x, review, color="#2a78d6", width=0.6)
    ax.set_ylim(0, max(0.5, review.max() * 1.15))
    ref_line(ax, MAX_REVIEW_SHARE, f"alert\n{MAX_REVIEW_SHARE:.0%}")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    no_run_marks(ax)
    ax.set_xticks(x, labels)

    # 4 Accuracy on images whose true class is known
    ax = fig.add_subplot(grid[2, 1])
    style(ax, "Accuracy on checked images")
    if "accuracy" in nights and nights["accuracy"].notna().any():
        acc = nights["accuracy"]
        ax.plot(x, acc, color="#2a78d6", linewidth=2, marker="o", markersize=5)
        ax.set_ylim(min(0.5, acc.min() - 0.05), 1.02)
        ref_line(ax, test_acc, f"test set\n{test_acc:.1%}")
        ref_line(ax, test_acc - MAX_ACCURACY_DROP, f"alert\n{test_acc - MAX_ACCURACY_DROP:.1%}")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    else:
        ax.text(0.5, 0.5, "No true classes known yet", ha="center", color=MUTED,
                transform=ax.transAxes)
    no_run_marks(ax)
    ax.set_xticks(x, labels)

    fig.text(0.05, 0.015, "Source: logs/run_summary.csv, logs/api.log, results/, "
             f"data/delivered_labels.csv · created {datetime.now():%d %b %Y %H:%M}",
             fontsize=7, color=MUTED)
    file.parent.mkdir(exist_ok=True)
    fig.savefig(file, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# --- Putting it together ------------------------------------------------------

def run_monitoring(now=None):
    """Read everything, check it, write the outputs. Returns the alerts."""
    setup_logging()
    now = now or datetime.now()
    REPORT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    runs = read_runs()
    api_log = read_api_log()
    predictions = add_true_labels(read_predictions())
    train_shares = training_class_shares()
    test_acc = test_accuracy()

    metrics = nightly_metrics(runs, predictions, api_log)
    alerts = check_alerts(runs, metrics, api_log, now, train_shares, test_acc)
    queues = write_review_queues(predictions)
    thresholds = threshold_table(predictions)
    if not metrics.empty:
        metrics.to_csv(REPORT_DIR / "nightly_metrics.csv", index=False)
    drawn = draw_dashboard(metrics, alerts, train_shares, test_acc,
                           REPORT_DIR / "dashboard.png") if not metrics.empty else False

    for level, message in alerts:
        log.log(logging.CRITICAL if level == "CRITICAL" else logging.WARNING, message)
    if not alerts:
        log.info("OK - no alerts")

    text = [f"Monitoring report, {now:%Y-%m-%d %H:%M}", ""]
    text += ["ALERTS"] + ([f"  [{lvl}] {msg}" for lvl, msg in alerts] or ["  none - all checks passed"])
    if not metrics.empty:
        cols = [c for c in ["night", "status", "runs", "runs_not_ok", "predicted", "failed_images",
                            "left_in_incoming", "batch_seconds", "api_errors",
                            "slowest_request_ms", "mean_confidence", "review_share",
                            "labelled", "accuracy"] if c in metrics]
        text += ["", "NIGHTS (last 14)", metrics[cols].tail(14).to_string(index=False)]
    if not thresholds.empty:
        text += ["", f"REVIEW THRESHOLD (all checked images so far; in use: {REVIEW_THRESHOLD})",
                 thresholds.to_string(index=False)]
    if queues:
        name, size = queues[-1]
        text += ["", f"REVIEW QUEUE: results/{name} ({size} images to check)"]
    text += ["", "Dashboard: reports/dashboard.png" if drawn else "No dashboard yet (no predictions)"]
    report = "\n".join(text)
    (REPORT_DIR / "monitoring_report.txt").write_text(report + "\n")
    print(report)
    return alerts


if __name__ == "__main__":
    found = run_monitoring()
    # 0 = all fine, 1 = at least one alert (useful for cron or a later e-mail step)
    sys.exit(1 if found else 0)