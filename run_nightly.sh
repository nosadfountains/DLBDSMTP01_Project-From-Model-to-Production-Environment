#!/bin/bash
# Nightly run (WP6): start the API, run the batch job, stop the API.
#
# cron calls this script every night. You can also start it by hand
# ("run now") from the project folder:   ./run_nightly.sh
#
# If an API is already running on port 8000, it is used and left running.

cd "$(dirname "$0")" || exit 1          # go to the project folder
PYTHON=".venv/bin/python"               # the project's own Python (cron knows no venv)
API_URL="http://127.0.0.1:8000"
mkdir -p logs

echo "=== Nightly run $(date '+%Y-%m-%d %H:%M:%S') ==="

started_api=no
if ! curl -s -o /dev/null "$API_URL/health"; then
    echo "Starting the API ..."
    "$PYTHON" src/api.py >> logs/api_console.log 2>&1 &
    api_pid=$!
    started_api=yes
    # Wait up to 60 seconds until the API answers
    for _ in $(seq 1 60); do
        curl -s -o /dev/null "$API_URL/health" && break
        sleep 1
    done
fi

"$PYTHON" src/batch_predict.py
status=$?

if [ "$started_api" = yes ]; then
    echo "Stopping the API ..."
    kill "$api_pid"
fi

echo "=== Finished with exit code $status ==="
exit $status