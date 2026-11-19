# Evidence from the running system

Real outputs from the system on my Mac. They back up the slides; the code that produced them is in `src/`.

| File | Shows | Slide |
|---|---|---|
| `api_health.json`, `api_model_info.json` | The service is up and which model it serves | 8 Model as a service |
| `api_predict.json` | A request with two photos and the answer: top class, confidence, probability of every class | 8 Model as a service |
| `api_error_empty.txt` | Error handling: a request without images gets a clear 400 message | 8 Model as a service |
| `api_error_mixed.json` | Error handling: an unreadable file is marked as error, the other photo is still predicted | 8 Model as a service |
| `crontab.txt` | The schedule: arrivals at 18:00, nightly run at 22:15 | 9 Nightly batch pipeline |
| `nightly_run.txt` | One scheduled evening without manual help, from arrival to finished run | 9 and 11 |
| `run_summary.csv` | Every batch run so far: status, photos, duration | 9 Nightly batch pipeline |
| `predictions_excerpt.csv` | First 10 rows of a results file for the refund team | 11 Results in a user scenario |
| `review_queue_excerpt.csv` | The 10 most unsure photos of one night, for a person to check | 10 Monitoring, 11 |
| `dashboard.png`, `monitoring_report.txt` | Monitoring after the first two nights: no alerts | 10 Monitoring |

The model metrics and the confusion matrix are in `docs/model_report_v1.txt` and `docs/confusion_matrix_v1.png` (slide 7).

Note: the second night was started by hand ("run now"), not by cron; the first evening (18:00 and 22:15) ran on schedule.
