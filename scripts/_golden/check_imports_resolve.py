"""Statically resolve every first-party import in the repo. No module is executed.

    python scripts/_golden/check_imports_resolve.py

Parses each .py with `ast` and, for every `import X` / `from X import ...` whose root package is
first-party, resolves the dotted path against the FILESYSTEM: `a.b.c` must be `a/b/c/` or `a/b/c.py`
under the repo root.

Why not `importlib.util.find_spec`, which is the obvious way to do this: `find_spec("a.b.c")`
imports the parent packages `a` and `a.b` to ask them for their `__path__`. Only the leaf is spared.
That was fatal when app/site_analysis/features/** existed -- some of its modules fired live HTTP/LLM
calls at module scope. That tree is gone, but the same hazard applies to proforma/backtests/**, so
the filesystem check stays: it cannot execute anything, by construction.

Why this exists: the 2026-07 restructure renamed app/utils -> app/core and the sweep missed
datafetching/ (now experiments/datafetching/), which is in none of import_smoke.py's TREES. Five modules there imported
app.utils and were broken for two commits before an audit caught it. Exit 1 on any unresolved
first-party import.
"""
from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# Every top-level package this repo owns. `experiments` is here because proforma/ui imports
# experiments.council.streamlit_view, and the council imports app.pnl_analysis.insights back.
FIRST_PARTY = ("app", "proforma", "libs", "experiments", "scripts")

# Not our problem: vendored trees and the venv. `experiments` is deliberately NOT in this set --
# see the note below.
SKIP_DIRS = {"venv", ".claude", ".git", "__pycache__", "archive", "node_modules"}

# A trap this checker fell into, recorded so it is not re-set: it once skipped `experiments/` and
# listed `datafetching` (not `experiments`) as first-party. Both were true before datafetching/ was
# quarantined into experiments/. After the move, the exact regression this file was written to catch
# -- a rename breaking experiments/datafetching's `app.*` imports -- became invisible to it, and the
# `experiments.council` import in proforma/ui/panels/_explore_markets.py was never first-party at
# all, so a tree with no council/ still reported "ok". Moving a directory can disarm the guard that
# watches it. If you quarantine a tree, check what stopped looking at it.

# Known-broken imports, recorded rather than hidden. Empty since 2026-07: test_endpoint.py, the only
# entry, imported two symbols that never existed, and was deleted with the site_analysis subsystem.
KNOWN_BROKEN: set = set()


def first_party(mod: str) -> bool:
    return mod.split(".")[0] in FIRST_PARTY


def resolves(dotted: str) -> bool:
    """True iff `a.b.c` maps to REPO/a/b/c/ or REPO/a/b/c.py. Touches only the filesystem.

    A bare directory counts: `app/`, `proforma/` and `proforma/` are PEP 420 namespace
    packages with no __init__.py, and `from proforma.models import coldstart` resolves
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
