"""Statically resolve every first-party import in the repo. No module is executed.

    python scripts/_golden/check_imports_resolve.py

Parses each .py with `ast` and, for every `import X` / `from X import ...` whose root package is
first-party (app, proforma, libs), asks importlib whether the dotted path resolves. `find_spec`
walks the finders without running module code, so this is safe for the trees that fire live
HTTP/LLM calls at import (app/site_analysis/features/**, proforma/v1_5/backtests/**).

Why this exists: the 2026-07 restructure renamed app/utils -> app/core and the sweep missed
datafetching/, which is not in any of import_smoke.py's TREES. Five modules there imported
app.utils and were broken for two commits before an audit caught it. `find_spec` over the whole
repo is cheap and would have caught it immediately. Exit 1 on any unresolved first-party import.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FIRST_PARTY = ("app", "proforma", "libs")
# Not our problem: vendored/legacy trees, and the venv.
SKIP_DIRS = {"venv", ".claude", ".git", "__pycache__", "archive", "experiments", "node_modules"}

# Known-broken, recorded rather than hidden. test_endpoint.py is a root ad-hoc script (not pytest)
# that imports `get_competitors_dynamics_endpoint` and `CompetitorsDynamicsRequest`. NEITHER SYMBOL
# EXISTS in app/, and `git grep` at the pre-refactor tag shows neither existed then -- the script has
# been broken for a long time. The restructure split routes.py/models.py into router/schemas/service,
# so its failure moved from "module imports, symbol missing" to "module missing". Broken either way.
# See docs/DIVERGENCES.md section 7. Delete this entry the day someone fixes or deletes the script.
KNOWN_BROKEN = {
    ("test_endpoint.py", "app.site_analysis.server.routes"),
    ("test_endpoint.py", "app.site_analysis.server.models"),
}


def first_party(mod: str) -> bool:
    return mod.split(".")[0] in FIRST_PARTY


def main() -> int:
    sys.path.insert(0, str(REPO))
    bad: list[tuple[str, int, str]] = []
    checked = 0

    for p in sorted(REPO.rglob("*.py")):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError:
            continue  # import_smoke.py owns syntax errors
        rel = p.relative_to(REPO)
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if not first_party(m):
                    continue
                checked += 1
                try:
                    found = importlib.util.find_spec(m) is not None
                except (ImportError, AttributeError, ValueError):
                    found = False
                if not found and (str(rel), m) not in KNOWN_BROKEN:
                    bad.append((str(rel), node.lineno, m))

    if bad:
        print(f"FAIL  {len(bad)} unresolved first-party import(s) of {checked} checked:")
        for f, ln, m in bad:
            print(f"        {f}:{ln}  ->  {m}")
        return 1
    print(f"   ok  {checked} first-party imports resolve "
          f"({len(KNOWN_BROKEN)} known-broken allowed; static, nothing executed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
