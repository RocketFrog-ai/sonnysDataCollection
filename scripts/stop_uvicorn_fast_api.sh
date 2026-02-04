if [ -z $RFW_HOME ] ; then
  echo "RFW_HOME not set, run: source .env"
  exit 2;
fi


if [ -z "$ENV_NAME" ]; then
  echo "❌ Usage: $0 <ENV_NAME>"
  echo "Example: $0 DEV"
  exit 1
fi

PID_FILE="$RFW_HOME/sonnysDataCollection/fastapi_$ENV_NAME.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "⚠️  No PID file found for $ENV_NAME ($PID_FILE). Nothing to stop."
  exit 0
fi

PID=$(cat "$PID_FILE")
echo $PID

# Check if the process is still running
if ps -p $PID > /dev/null 2>&1; then
  echo "🛑 Stopping Uvicorn Fast API ($ENV_NAME) with PID $PID ..."
  kill $PID
  sleep 2
  if ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Process $PID did not stop gracefully. Forcing kill..."
    kill -9 $PID
  fi
  rm -f "$PID_FILE"
  echo "✅ Uvicorn Fast API for ($ENV_NAME) Env stopped successfully."
else
  echo "⚠️  No running process found with PID $PID. Removing stale PID file."
  rm -f "$PID_FILE"
fi
