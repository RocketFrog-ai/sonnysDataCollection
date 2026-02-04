./stop_uvicorn_fast_api.sh
sleep 5

if [ -z $RFW_HOME ] ; then
  echo "RFW_HOME not set, run: source .env"
  exit 2;
fi

if [ -z $CONDA_ENV_NAME ] ; then
  echo "CONDA_ENV_NAME not set, run: source .env"
  exit 2;
fi

currenv=`conda info --env | grep "*"|awk ' { print $1 } '`
echo "currenv - $currenv"
if [ $currenv == $CONDA_ENV_NAME ]
then
    echo "Conda Environment Verified Successfully ..."
else
    echo "ERROR: Conda Environment Verification Failed. Run 'conda activate $CONDA_ENV_NAME'"
    exit 1;
fi


cd $RFW_HOME/sonnysDataCollection/

bs_log_fn="proforma-fastapi-"`date +"%d-%b-%Y-%H-%M-%S"`".log";echo $curr_ts
bs_log_pfn="$RFW_HOME/logs/""$bs_log_fn"
echo "start_uvicorn_fast_api.sh - MESSAGE: Starting the server now logs available in: $bs_log_pfn"

# nohup python -m ca.server.apis --env_name $ENV_NAME >$bs_log_pfn 2>&1 &
nohup python -m app.server.main >$bs_log_pfn 2>&1 &
echo $! > "fastapi_$ENV_NAME.pid"