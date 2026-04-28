#!/usr/bin/env bash
# Run clustering_v2 Streamlit on port 8080 (bind all interfaces).
cd "$(dirname "$0")/../../../.." || exit 1
exec python -m streamlit run daily_data/daily-data-modelling/clustering_v2/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8080 \
  --server.headless true \
  "$@"
