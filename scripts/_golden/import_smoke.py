"""Syntax + import smoke over the live trees. Records the PASS/FAIL set as a golden.

    python scripts/_golden/import_smoke.py <out_dir> <tag>       # tag: backend | ui

Philosophy: some modules already fail to import today (feature packages that rely on the
startup scripts putting their dir on PYTHONPATH). We do not fix those here. Instead we
freeze the exact pass/fail set, so the refactor is required to change it in no way at all.
A module that starts failing -- or one that starts passing for a surprising reason -- shows
up as a diff.

STREAMLIT COVERAGE IS DELIBERATELY PARTIAL. Importing a Streamlit entrypoint executes the
whole script top to bottom (it is a script, not a library), so we `ast.parse` every UI file
but only *import* the non-entrypoint modules. This is a real gap, not an oversight: the UI
has no golden output, and this file is where that is written down rather than papered over.
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
UI_ENTRYPOINTS = {"app.py", "site_analysis_page.py", "site_visual_page.py", "streamlit_view.py"}

# Trees that are SCRIPTS, not libraries: they run work at module level. Importing
# app/site_analysis/features/** actually fires live LLM/HTTP calls (typeOfSite/test_o4_mini.py
# hits an API and prints `Response: 403`; o4mini_images_classification.py calls a vision model
# at module scope) and one calls sys.exit(). They are out of this refactor's scope by decision:
# the startup scripts put their dirs on PYTHONPATH and they use bare intra-feature imports.
# We syntax-check them and never import them. This narrows coverage, on purpose, in writing.
AST_ONLY_PREFIXES = ("app/site_analysis/features/",)

TREES = {
    # tag -> (roots to walk, roots to import-as-package)
    "backend": ["app"],
    "ui": ["earnest-proforma-2.0/streamlits", "proforma/v1_5/ui", "proforma/v1_5/models",
           "proforma/v1_6", "council"],
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
    result = {"_tag": tag, "_python": sys.version.split()[0], "ast": {}, "import": {}}

    for root_s in TREES[tag]:
        root = REPO / root_s
        for p in py_files(root):
            rel = str(p.relative_to(REPO))
            # 1) syntax check, always
            try:
                ast.parse(p.read_text(encoding="utf-8"), filename=rel)
                result["ast"][rel] = "ok"
            except SyntaxError as e:
                result["ast"][rel] = f"SyntaxError: {e.msg} (line {e.lineno})"
                continue

            # 2) import, where safe
            if rel.startswith(AST_ONLY_PREFIXES):
                result["import"][rel] = "skipped: script tree, side effects at import (see AST_ONLY_PREFIXES)"
                continue
            if p.name in UI_ENTRYPOINTS:
                result["import"][rel] = "skipped: streamlit entrypoint (executes on import)"
                continue
            name = mod_name(p)
            if not all(part.isidentifier() for part in name.split(".")):
                result["import"][rel] = f"skipped: '{name}' is not an importable dotted path"
                continue
            try:
                importlib.import_module(name)
                result["import"][rel] = "ok"
            except BaseException as e:  # noqa: BLE001 - some modules raise SystemExit
                result["import"][rel] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}" if str(e) else type(e).__name__

    dest = Path(out_dir) / f"imports_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    ok = sum(1 for v in result["import"].values() if v == "ok")
    skip = sum(1 for v in result["import"].values() if v.startswith("skipped"))
    fail = len(result["import"]) - ok - skip
    bad_ast = sum(1 for v in result["ast"].values() if v != "ok")
    print(f"[import_smoke:{tag}] ast {len(result['ast'])} files ({bad_ast} bad) | "
          f"import ok={ok} skipped={skip} failed={fail} -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/_refactor/baseline",
         sys.argv[2] if len(sys.argv) > 2 else "backend")
