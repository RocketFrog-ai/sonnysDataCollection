./stop_uvicorn_fast_api.sh
sleep 5

# Load .env so RFW_HOME and CONDA_ENV_NAME are set (e.g. when script is run without prior source .env)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/../.env" ] && source "${SCRIPT_DIR}/../.env"

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


cd $RFW_HOME/sonnysDataCollection/

# So "from utils.xxx" in app/features/competitors and "from nearbyStores.xxx" in app/features/nearbyStores resolve
export PYTHONPATH="${RFW_HOME}/sonnysDataCollection/app/features/competitors:${RFW_HOME}/sonnysDataCollection/app/features${PYTHONPATH:+:$PYTHONPATH}"

bs_log_fn="proforma-fastapi-"`date +"%d-%b-%Y-%H-%M-%S"`".log";echo $curr_ts
bs_log_pfn="$RFW_HOME/logs/""$bs_log_fn"
echo "start_uvicorn_fast_api.sh - MESSAGE: Starting the server now logs available in: $bs_log_pfn"

# nohup python -m ca.server.apis --env_name $ENV_NAME >$bs_log_pfn 2>&1 &
PYTHONPATH="${RFW_HOME}/sonnysDataCollection/app/features/competitors:${RFW_HOME}/sonnysDataCollection/app/features${PYTHONPATH:+:$PYTHONPATH}" nohup python -m app.server.main >"$bs_log_pfn" 2>&1 &
echo $! > "fastapi_$ENV_NAME.pid"
