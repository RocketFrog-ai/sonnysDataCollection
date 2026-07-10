#!/usr/bin/env bash
# Behavior-preservation smoke test for the repo restructure.
#
#   scripts/smoke.sh                    verify current tree against the golden baseline
#   scripts/smoke.sh --capture-baseline (re)write the baseline; only legitimate pre-refactor
#
# It checks, in the correct conda env for each component:
#   1. the joblib artifact unpickles in the BACKEND env, un-refit      (the version-sensitive one)
#   2. libs/carwash_type imports (no golden test covers it; see the note at the step)
#   3. every first-party import in the repo resolves, statically       (nothing executed)
#   4. coldstart predict_site / predict_neighbours / cannib_params     (24 cases, 3 real pins)
#   5. every deterministic /v1/pnl_analysis/* endpoint, in-process     (15 cases)
#   6. the Streamlit app body actually executes (AppTest), widget surface frozen
#   7. ast-parse + import smoke over app/ and the UI/model trees       (pass/fail set frozen)
# then diffs 4-7 against scripts/_golden/baseline/ at 1e-9.
#
# WHAT THIS DOES NOT COVER, honestly:
#   * The Streamlit UI is executed (step 6, via AppTest) but only its FIRST render pass. Widgets
#     that appear after picking a mode / dropping a pin / clicking are never exercised, and nothing
#     compares pixels. A layout or interaction regression will NOT be caught here.
#   * /v1/pnl_analysis/insights/* are excluded: they call an LLM and are non-deterministic.
#   * app/site_analysis/features/** is ast-parsed but never imported -- those modules run live
#     HTTP/LLM calls at module scope. See scripts/_golden/import_smoke.py:AST_ONLY_PREFIXES.
#   * app/site_analysis/* endpoints have no golden outputs -- they make live third-party calls.
#     Their route paths and OpenAPI schema were diffed against the pre-refactor tag by hand, once.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/homebrew/Caskroom/miniconda/base)"
PY_BACKEND="$CONDA_BASE/envs/sonnysDataCollection/bin/python"   # py3.9  — FastAPI + the joblib artifact
PY_UI="$CONDA_BASE/envs/proforma311/bin/python"                 # py3.11 — Streamlit

BASELINE="scripts/_golden/baseline"
CAPTURE=0
[[ "${1:-}" == "--capture-baseline" ]] && CAPTURE=1
OUT="$([[ $CAPTURE == 1 ]] && echo "$BASELINE" || mktemp -d)"

for p in "$PY_BACKEND" "$PY_UI"; do
  [[ -x "$p" ]] || { echo "FATAL: missing interpreter $p"; echo "  create it: conda env create -f environment*.yml"; exit 2; }
done

# Hazard 4: a deleted module keeps importing from its .pyc. Purge before every pass.
find . -name __pycache__ -not -path './venv/*' -not -path './.claude/*' -exec rm -rf {} + 2>/dev/null || true

echo "== 1/8 joblib artifact unpickles in the BACKEND env (un-refit) =="
"$PY_BACKEND" - <<'PY'
import glob, joblib, sklearn
hits = glob.glob("proforma/v1_5/artifacts/coldstart_artifacts.joblib") or \
       glob.glob("earnest-proforma-2.0/notebooks/artifacts/coldstart_artifacts.joblib")  # pre-refactor fallback
assert hits, "coldstart_artifacts.joblib not found in either location"
a = joblib.load(hits[0])
assert isinstance(a, dict) and "models" in a and "ramps" in a, "artifact shape changed"
print(f"   ok  {hits[0]}  ({len(a)} keys, sklearn {sklearn.__version__})")
PY

# Assertion, not a golden diff. libs/carwash_type's config.py resolves .env by walking up from
# __file__; a move at the wrong depth makes load_dotenv() no-op silently and the module raises, and
# nothing else would catch it. NOTE (2026-07): its only importer,
# features/active/nearbyCompetitors/classify_competitor_types.py, became unreachable when the Celery
# pipeline was removed. The package is now reached only by its own CLI (classify_site_types.py).
# Kept as a check because the CLI is real; drop this step if the package goes.
echo "== 2/8 libs/carwash_type imports (CLI-only since the Celery removal) =="
"$PY_BACKEND" - <<'PY'
import importlib, sys
sys.path.insert(0, ".")
for m in ("libs.carwash_type.config", "libs.carwash_type.scraper",
          "libs.carwash_type.analyzer", "libs.carwash_type.finder"):
    importlib.import_module(m)
print("   ok  4 modules import; .env resolved from repo root")
PY

# Whole-repo static import resolution. import_smoke.py only walks app/ and the proforma trees, so
# it never saw datafetching/ -- which is exactly where the app/utils -> app/core rename broke five
# modules for two commits. Resolution is done against the FILESYSTEM, not importlib: find_spec()
# imports a module's parent packages to ask for their __path__, which would execute
# app/site_analysis/features/**, the tree we promise never to import. See the checker's docstring.
echo "== 3/8 every first-party import resolves, repo-wide (static) =="
"$PY_BACKEND" scripts/_golden/check_imports_resolve.py 2>&1 | tail -1

echo "== 4/8 coldstart golden (backend env) =="
"$PY_BACKEND" scripts/_golden/capture_model.py "$OUT" 2>&1 | grep -v "^\s*$" | tail -1

echo "== 5/8 pnl_analysis API golden (backend env) =="
"$PY_BACKEND" scripts/_golden/capture_api.py "$OUT" 2>&1 | tail -1

echo "== 6/8 streamlit app body executes (AppTest, streamlit env) =="
"$PY_UI" scripts/_golden/capture_ui.py "$OUT" 2>&1 | grep -viE "InconsistentVersion|scikit-learn|warnings.warn|ScriptRunContext|st.components" | tail -1

echo "== 7/8 import smoke =="
"$PY_BACKEND" scripts/_golden/import_smoke.py "$OUT" backend 2>&1 | tail -1
"$PY_UI"      scripts/_golden/import_smoke.py "$OUT" ui      2>&1 | tail -1

if [[ $CAPTURE == 1 ]]; then
  echo; echo "BASELINE CAPTURED -> $BASELINE"; exit 0
fi

echo "== 8/8 diff vs baseline (tol 1e-9) =="
rc=0
# model.json / api.json are the behavior contract: keyed by logical case name, frozen forever.
# ui_render.json is the app's first-pass widget surface, also keyed by logical name.
# imports_*.json are keyed by file path, so their strict surface is only the (empty) failure
# maps; check_counts.py guards the informational counts against silent regression.
for f in model.json api.json ui_render.json imports_backend.json imports_ui.json; do
  "$PY_BACKEND" scripts/_golden/diff.py "$BASELINE/$f" "$OUT/$f" 1e-9 || rc=1
done
for t in backend ui; do
  "$PY_BACKEND" scripts/_golden/check_counts.py "$BASELINE/imports_$t.json" "$OUT/imports_$t.json" || rc=1
done
rm -rf "$OUT"
[[ $rc == 0 ]] && echo && echo "SMOKE PASS -- behavior preserved to 1e-9" || { echo; echo "SMOKE FAIL"; }
exit $rc
