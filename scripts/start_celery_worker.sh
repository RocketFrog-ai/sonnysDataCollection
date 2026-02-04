./stop_celery_worker.sh
sleep 5

echo $RFW_HOME

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

cd ../../

mkdir -p logs
mkdir -p data

cd $RFW_HOME/sonnysDataCollection/

bs_log_fn="proforma-celery-"`date +"%d-%b-%Y-%H-%M-%S"`".log";echo $curr_ts
bs_log_pfn="$RFW_HOME/logs/""$bs_log_fn"
echo "start_celery_worker.sh - MESSAGE: Starting the server now logs available in: $bs_log_pfn"
nohup celery -A app.celery.celery_app worker --loglevel=info --concurrency=2 >$bs_log_pfn 2>&1 &
echo $! > "celery_$ENV_NAME.pid"