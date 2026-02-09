./stop_celery_worker.sh
sleep 5

# Load .env so RFW_HOME and CONDA_ENV_NAME are set (e.g. when script is run without prior source .env)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/../.env" ] && source "${SCRIPT_DIR}/../.env"

echo $RFW_HOME

if [ -z "$RFW_HOME" ] ; then
  echo "RFW_HOME not set, run: source .env"
  exit 2;
fi

if [ -z "$CONDA_ENV_NAME" ] ; then
  echo "CONDA_ENV_NAME not set, run: source .env"
  exit 2;
fi

currenv="${CONDA_DEFAULT_ENV:-}"
echo "currenv - $currenv"
if [ "$currenv" = "$CONDA_ENV_NAME" ]
then
    echo "Conda Environment Verified Successfully ..."
else
    echo "ERROR: Conda Environment Verification Failed. Run 'conda activate $CONDA_ENV_NAME'"
    exit 1;
fi

cd ../../

mkdir -p logs
mkdir -p data

REPO_ROOT="$RFW_HOME/sonnysDataCollection"
cd "$REPO_ROOT" || exit 3

# So "from utils.xxx" in app/features/competitors and "from nearbyStores.xxx" in app/features/nearbyStores resolve
export PYTHONPATH="${REPO_ROOT}/app/features/competitors:${REPO_ROOT}/app/features${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$REPO_ROOT/logs"
bs_log_fn="proforma-celery-"`date +"%d-%b-%Y-%H-%M-%S"`".log";echo $curr_ts
bs_log_pfn="$REPO_ROOT/logs/$bs_log_fn"
echo "start_celery_worker.sh - MESSAGE: Starting the server now logs available in: $bs_log_pfn"
PYTHONPATH="${REPO_ROOT}/app/features/competitors:${REPO_ROOT}/app/features${PYTHONPATH:+:$PYTHONPATH}" nohup "${CONDA_PREFIX}/bin/celery" -A app.celery.celery_app worker --loglevel=info --concurrency=2 >"$bs_log_pfn" 2>&1 &
echo $! > "$REPO_ROOT/celery_$ENV_NAME.pid"
