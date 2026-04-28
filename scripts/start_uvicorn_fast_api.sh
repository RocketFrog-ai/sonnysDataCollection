#!/bin/bash

./stop_uvicorn_fast_api.sh
sleep 5

# Load .env so RFW_HOME and CONDA_ENV_NAME are set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/../.env" ] && source "${SCRIPT_DIR}/../.env"

if [ -z "$RFW_HOME" ]; then
  echo "RFW_HOME not set, run: source .env"
  exit 2
fi

if [ -z "$CONDA_ENV_NAME" ]; then
  echo "CONDA_ENV_NAME not set, run: source .env"
  exit 2
fi

# Clean Conda env safely
currenv=$(printf "%s\n" "$CONDA_DEFAULT_ENV" | grep -v '^#' | tail -n 1 | xargs)

echo "currenv - $currenv"

if [ "$currenv" = "$CONDA_ENV_NAME" ]; then
    echo "Conda Environment Verified Successfully ..."
else
    echo "ERROR: Conda Environment Verification Failed. Run 'conda activate $CONDA_ENV_NAME'"
    exit 1
fi

REPO_ROOT="$RFW_HOME/sonnysDataCollection"
cd "$REPO_ROOT" || exit 3

# Ensure imports resolve
export PYTHONPATH="${REPO_ROOT}/app/site_analysis/features/competitors:${REPO_ROOT}/app/site_analysis/features${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPO_ROOT/logs"

bs_log_fn="proforma-fastapi-$(date +"%d-%b-%Y-%H-%M-%S").log"
bs_log_pfn="$REPO_ROOT/logs/$bs_log_fn"
echo "start_uvicorn_fast_api.sh - MESSAGE: Starting the server now logs available in: $bs_log_pfn"

nohup python -m app.site_analysis.server.main > "$bs_log_pfn" 2>&1 &
echo $! > "$REPO_ROOT/fastapi_${ENV_NAME}.pid"

echo "FastAPI server started successfully with PID $(cat "$REPO_ROOT/fastapi_${ENV_NAME}.pid")"