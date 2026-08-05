# Review Analysis — self-contained image.
#
# Everything the app needs is in this folder: the review CSV, the precomputed
# AI answers, the theme config. The only thing that must come from outside is
# the Azure OpenAI key, which is read from the environment at runtime and is
# deliberately never baked into a layer (see .dockerignore).
#
#   docker build -t review-analysis .
#   docker run -p 8501:8501 -e AZURE_OPENAI_API_KEY=... review-analysis
#
# Without a key the dashboard works in full; only the AI-insights button
# reports that it has nothing to call.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Streamlit inside a container: bind all interfaces, no browser, no
    # usage stats, and no file watcher (the code never changes in an image).
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unprivileged. The app writes one file at runtime —
# data/ai_insights_cache.json — so that directory must be owned by the user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit's own health endpoint; the container is unhealthy until it serves.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.getenv('PORT','8501'); \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/_stcore/health', timeout=4).status == 200 else 1)"

# Listen on whatever port the platform injects (Azure App Service, Cloud Run
# and Fly all set PORT), falling back to 8501 for a plain `docker run`.
CMD ["sh", "-c", "streamlit run app/Home.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
