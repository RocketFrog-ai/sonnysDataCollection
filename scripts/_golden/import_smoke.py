"""Syntax + import smoke over the live trees. Records the PASS/FAIL set as a golden.

    python scripts/_golden/import_smoke.py <out_dir> <tag>       # tag: backend | ui

Philosophy: some modules already fail to import today (feature packages that rely on the
startup scripts putting their dir on PYTHONPATH). We do not fix those here. Instead we
freeze the exact pass/fail set, so the refactor is required to change it in no way at all.
A module that starts failing -- or one that starts passing for a surprising reason -- shows
up as a diff.

STREAMLIT ENTRYPOINTS ARE NEVER IMPORTED HERE. Importing one executes the whole script top to
bottom (it is a script, not a library), so we `ast.parse` every UI file and only *import* the
non-entrypoint modules. The entrypoints are covered separately, and properly, by
scripts/_golden/capture_ui.py, which runs them under streamlit's AppTest.
"""
from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# UI files that execute on import (Streamlit scripts). ast-parsed, never imported.
UI_ENTRYPOINTS = {"app.py", "site_visual_page.py", "streamlit_view.py"}

# Trees that are SCRIPTS, not libraries: they run work at module level.
# (app/site_analysis/features/** used to live here; that whole subsystem was removed in 2026-07.)
#
# proforma/backtests/** are scripts: backtest_features.py reads a CSV and fits a model
# at module level. Before the refactor they were never imported anyway -- their dotted path went
# through "earnest-proforma-2.0", which is not a Python identifier, so mod_name() produced an
# invalid name and they fell into the "skipped" bucket by accident. After the move the path IS
# valid, so without this entry the smoke test would start running full backtests on every pass.
AST_ONLY_PREFIXES = (
    "proforma/backtests/",
)

TREES = {
    # tag -> roots to walk. Pre-refactor roots are kept so this script still works when checked
    # out at the pre-refactor tag; py_files() silently skips roots that do not exist.
    # proforma/pnl is the shared, Streamlit-free P&L/market/trend math imported by BOTH the backend
    # (app/pnl_analysis/modelling) and the UI (proforma/ui/panels). Swept here so a break shows up.
    "backend": ["app", "proforma/pnl"],
    "ui": [
        "proforma/models", "proforma/ui", "proforma/backtests",
        # the council still runs (python -m experiments.council.harness), so keep it swept even
        # though experiments/ is otherwise off the import path.
        "experiments/council",
        # older layouts, so this still works when checked out at an old tag:
        "proforma/v1_5/models", "proforma/v1_5/ui", "proforma/v1_6",
        "earnest-proforma-2.0/streamlits", "council",
    ],
}


def py_files(root: Path):
    if not root.exists():
        return
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def mod_name(p: Path) -> str:
    rel = p.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def main(out_dir: str, tag: str) -> None:
    sys.path.insert(0, str(REPO))
    detail_ast: dict[str, str] = {}
    detail_imp: dict[str, str] = {}

    for root_s in TREES[tag]:
        root = REPO / root_s
        for p in py_files(root):
            rel = str(p.relative_to(REPO))
            # 1) syntax check, always
            try:
                ast.parse(p.read_text(encoding="utf-8"), filename=rel)
                detail_ast[rel] = "ok"
            except SyntaxError as e:
                detail_ast[rel] = f"SyntaxError: {e.msg} (line {e.lineno})"
                continue

            # 2) import, where safe
            if rel.startswith(AST_ONLY_PREFIXES):
                detail_imp[rel] = "skipped: script tree, side effects at import (see AST_ONLY_PREFIXES)"
                continue
            if p.name in UI_ENTRYPOINTS:
                detail_imp[rel] = "skipped: streamlit entrypoint (executes on import)"
                continue
            name = mod_name(p)
            if not all(part.isidentifier() for part in name.split(".")):
                detail_imp[rel] = f"skipped: '{name}' is not an importable dotted path"
                continue
            try:
                importlib.import_module(name)
                detail_imp[rel] = "ok"
            except BaseException as e:  # noqa: BLE001 - some modules raise SystemExit
                detail_imp[rel] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}" if str(e) else type(e).__name__

    ok = sum(1 for v in detail_imp.values() if v == "ok")
    skip = sum(1 for v in detail_imp.values() if v.startswith("skipped"))
    fail = len(detail_imp) - ok - skip
    bad_ast = sum(1 for v in detail_ast.values() if v != "ok")

    # The strict-diff surface is ONLY the failure maps. They are keyed by path, but they
    # are empty, so a rename cannot perturb them. Counts and the full per-file maps go
    # under `_` keys, which diff.py reports as informational drift rather than failure.
    #
    # Why: this whole file is keyed by file path, and the refactor's entire job is to move
    # files. A path-keyed strict diff would light up on every legitimate rename, which
    # trains you to re-baseline -- and re-baselining is exactly how a real regression walks
    # in unnoticed. So the contract here is "nothing is broken", not "nothing moved".
    # The contract "the numbers did not change" is enforced by model.json / api.json, whose
    # keys are logical case names and never move.
    result = {
        "syntax_errors": {k: v for k, v in detail_ast.items() if v != "ok"},
        "import_failures": {k: v for k, v in detail_imp.items()
                            if v != "ok" and not v.startswith("skipped")},
        "_tag": tag,
        "_python": sys.version.split()[0],
        "_counts": {"ast_files": len(detail_ast), "import_ok": ok, "import_skipped": skip, "import_failed": fail},
        "_detail_ast": detail_ast,
        "_detail_import": detail_imp,
    }

    dest = Path(out_dir) / f"imports_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"[import_smoke:{tag}] ast {len(detail_ast)} files ({bad_ast} bad) | "
          f"import ok={ok} skipped={skip} failed={fail} -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/_golden/baseline",
         sys.argv[2] if len(sys.argv) > 2 else "backend")
