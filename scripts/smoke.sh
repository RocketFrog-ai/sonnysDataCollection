#!/usr/bin/env bash
# Behavior-preservation smoke test for the repo restructure.
#
#   scripts/smoke.sh                    verify current tree against the golden baseline
#   scripts/smoke.sh --capture-baseline (re)write the baseline; only legitimate pre-refactor
#
# It captures, in the correct conda env for each component:
#   1. joblib artifact unpickles in the BACKEND env, un-refit          (the version-sensitive one)
#   2. coldstart predict_site / predict_neighbours / cannib_params     (24 cases, 3 real pins)
#   3. every deterministic /v1/pnl_analysis/* endpoint, in-process     (15 cases)
#   4. ast-parse + import smoke over app/ and the UI/model trees       (pass/fail set frozen)
# then diffs 2-4 against docs/_refactor/baseline/ at 1e-9.
#
# WHAT THIS DOES NOT COVER, honestly:
#   * The Streamlit UI has no golden output. A Streamlit entrypoint executes on import, so it is
#     only ast-parsed. A visual/behavioural regression in the UI will NOT be caught here.
#   * /v1/pnl_analysis/insights/* are excluded: they call an LLM and are non-deterministic.
#   * app/site_analysis/features/** is ast-parsed but never imported -- those modules run live
#     HTTP/LLM calls at module scope. See scripts/_golden/import_smoke.py:AST_ONLY_PREFIXES.
#   * The Celery async pipeline (POST /v1/analyze-site) is not exercised; it needs Redis.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/homebrew/Caskroom/miniconda/base)"
PY_BACKEND="$CONDA_BASE/envs/sonnysDataCollection/bin/python"   # py3.9  — FastAPI + Celery + the joblib artifact
PY_UI="$CONDA_BASE/envs/proforma311/bin/python"                 # py3.11 — Streamlit

BASELINE="docs/_refactor/baseline"
CAPTURE=0
[[ "${1:-}" == "--capture-baseline" ]] && CAPTURE=1
OUT="$([[ $CAPTURE == 1 ]] && echo "$BASELINE" || mktemp -d)"

for p in "$PY_BACKEND" "$PY_UI"; do
  [[ -x "$p" ]] || { echo "FATAL: missing interpreter $p"; echo "  create it: conda env create -f environment*.yml"; exit 2; }
done

# Hazard 4: a deleted module keeps importing from its .pyc. Purge before every pass.
find . -name __pycache__ -not -path './venv/*' -not -path './.claude/*' -exec rm -rf {} + 2>/dev/null || true

echo "== 1/6 joblib artifact unpickles in the BACKEND env (un-refit) =="
"$PY_BACKEND" - <<'PY'
import glob, joblib, sklearn
hits = glob.glob("proforma/v1_5/artifacts/coldstart_artifacts.joblib") or \
       glob.glob("earnest-proforma-2.0/notebooks/artifacts/coldstart_artifacts.joblib")
assert hits, "coldstart_artifacts.joblib not found in either location"
a = joblib.load(hits[0])
assert isinstance(a, dict) and "models" in a and "ramps" in a, "artifact shape changed"
print(f"   ok  {hits[0]}  ({len(a)} keys, sklearn {sklearn.__version__})")
PY

# Assertion, not a golden diff. libs/carwash_type is consumed by the live nearbyCompetitors
# feature, which lives under the ast-only features/ tree -- so no golden test covers it. Its
# config.py resolves .env by walking up from __file__; a move at the wrong depth makes
# load_dotenv() no-op silently and the module raises. Check it explicitly.
echo "== 2/6 libs/carwash_type imports (live nearbyCompetitors dependency) =="
"$PY_BACKEND" - <<'PY'
import importlib, sys
sys.path.insert(0, ".")
for m in ("libs.carwash_type.config", "libs.carwash_type.scraper",
          "libs.carwash_type.analyzer", "libs.carwash_type.finder"):
    importlib.import_module(m)
print("   ok  4 modules import; .env resolved from repo root")
PY

echo "== 3/6 coldstart golden (backend env) =="
"$PY_BACKEND" scripts/_golden/capture_model.py "$OUT" 2>&1 | grep -v "^\s*$" | tail -1

echo "== 4/6 pnl_analysis API golden (backend env) =="
"$PY_BACKEND" scripts/_golden/capture_api.py "$OUT" 2>&1 | tail -1

echo "== 5/6 import smoke =="
"$PY_BACKEND" scripts/_golden/import_smoke.py "$OUT" backend 2>&1 | tail -1
"$PY_UI"      scripts/_golden/import_smoke.py "$OUT" ui      2>&1 | tail -1

if [[ $CAPTURE == 1 ]]; then
  echo; echo "BASELINE CAPTURED -> $BASELINE"; exit 0
fi

echo "== 6/6 diff vs baseline (tol 1e-9) =="
rc=0
for f in model.json api.json imports_backend.json imports_ui.json; do
  "$PY_BACKEND" scripts/_golden/diff.py "$BASELINE/$f" "$OUT/$f" 1e-9 || rc=1
done
rm -rf "$OUT"
[[ $rc == 0 ]] && echo && echo "SMOKE PASS -- behavior preserved to 1e-9" || { echo; echo "SMOKE FAIL"; }
exit $rc
