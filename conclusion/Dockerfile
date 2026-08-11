# Conclusions demo — self-contained image.
#
# BUILD FROM THE REPO ROOT, not from conclusion/: the app resolves its data
# against the repository root (`REPO = Path(__file__).resolve().parents[2]`)
# and reads one file from proforma/data/opex, so both trees have to keep
# their relative layout inside the image.
#
#   az acr build --registry proformaacr --platform linux/amd64 \
#     --image conclusion-demo:latest -f conclusion/Dockerfile .
#
# The repo is ~1.7 GB (venv, archive, artifacts); the root .dockerignore is an
# allowlist that keeps the build context to the ~15 MB this app actually needs.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY conclusion/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only the five CSVs the sections read, plus the app itself.
# Theme first: the app reads theme.base at import time.
COPY conclusion/.streamlit/      ./.streamlit/
COPY conclusion/demo/            ./conclusion/demo/
COPY conclusion/data/historical_data_5yrs_monthly.csv \
     conclusion/data/historical_data_sitewise.csv \
     conclusion/data/n70_backtest_dataset.csv \
     conclusion/data/tunnel_length_with_wash.csv \
     ./conclusion/data/
COPY proforma/data/opex/opex-data.csv ./proforma/data/opex/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.getenv('PORT','8501'); \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "streamlit run conclusion/demo/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
