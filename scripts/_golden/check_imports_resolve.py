"""Statically resolve every first-party import in the repo. No module is executed.

    python scripts/_golden/check_imports_resolve.py

Parses each .py with `ast` and, for every `import X` / `from X import ...` whose root package is
first-party, resolves the dotted path against the FILESYSTEM: `a.b.c` must be `a/b/c/` or `a/b/c.py`
under the repo root.

Why not `importlib.util.find_spec`, which is the obvious way to do this: `find_spec("a.b.c")`
imports the parent packages `a` and `a.b` to ask them for their `__path__`. Only the leaf is spared.
That is fatal here, because `app/site_analysis/features/**` is a tree we promise never to import --
some of its modules fire live HTTP/LLM calls at module scope, and one calls sys.exit(). Concretely,
`find_spec("app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors")` executes
`nearbyCompetitors/__init__.py`, which re-exports from that very module, pulling seven modules under
features/ into sys.modules. A filesystem check cannot execute anything, by construction.

Why this exists: the 2026-07 restructure renamed app/utils -> app/core and the sweep missed
datafetching/, which is not in any of import_smoke.py's TREES. Five modules there imported
app.utils and were broken for two commits before an audit caught it. Exit 1 on any unresolved
first-party import.
"""
from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FIRST_PARTY = ("app", "proforma", "libs", "datafetching", "scripts")
# Not our problem: vendored/legacy trees, and the venv.
SKIP_DIRS = {"venv", ".claude", ".git", "__pycache__", "archive", "experiments", "node_modules"}

# Known-broken, recorded rather than hidden. test_endpoint.py is a root ad-hoc script (not pytest)
# that imports `get_competitors_dynamics_endpoint` and `CompetitorsDynamicsRequest`. NEITHER SYMBOL
# EXISTS in app/, and `git grep` at the pre-refactor tag shows neither existed then -- the script has
# been broken for a long time. The restructure split routes.py/models.py into router/schemas/service,
# so its failure moved from "module imports, symbol missing" to "module missing". Broken either way.
# See docs/DIVERGENCES.md section 8. Delete this entry the day someone fixes or deletes the script.
KNOWN_BROKEN = {
    ("test_endpoint.py", "app.site_analysis.server.routes"),
    ("test_endpoint.py", "app.site_analysis.server.models"),
}


def first_party(mod: str) -> bool:
    return mod.split(".")[0] in FIRST_PARTY


def resolves(dotted: str) -> bool:
    """True iff `a.b.c` maps to REPO/a/b/c/ or REPO/a/b/c.py. Touches only the filesystem.

    A bare directory counts: `app/`, `proforma/` and `proforma/v1_5/` are PEP 420 namespace
    packages with no __init__.py, and `from proforma.v1_5.models import coldstart` resolves
    through them at runtime.
    """
    p = REPO.joinpath(*dotted.split("."))
    return p.is_dir() or p.with_suffix(".py").is_file()


def main() -> int:
    bad: list[tuple[str, int, str]] = []
    checked = 0

    for p in sorted(REPO.rglob("*.py")):
        rel = p.relative_to(REPO)
        # Match the skip list against the REPO-RELATIVE path. Matching p.parts would test the
        # absolute path, so a checkout living under any dir named like a SKIP_DIRS entry (a git
        # worktree under .claude/, say) would silently skip every file and report 0 checked.
        if any(s in rel.parts for s in SKIP_DIRS):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError:
            continue  # import_smoke.py owns syntax errors
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
                if not resolves(m) and (str(rel), m) not in KNOWN_BROKEN:
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
