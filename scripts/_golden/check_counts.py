"""Guard the informational half of the import golden.

    python scripts/_golden/check_counts.py <baseline/imports_TAG.json> <cand/imports_TAG.json>

diff.py treats `_`-prefixed keys as informational so that file *renames* don't fail the
build. That leaves a hole: `_counts.import_ok` could quietly fall (a module stops being
discovered because a tree moved out from under TREES) without anything going red.

This closes it: import_ok may never DROP, and failures must stay empty.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    with open(sys.argv[1]) as f:
        base = json.load(f)
    with open(sys.argv[2]) as f:
        cand = json.load(f)

    tag = cand.get("_tag", "?")
    rc = 0

    for key in ("syntax_errors", "import_failures"):
        if cand.get(key):
            rc = 1
            print(f"FAIL  [{tag}] {key}: {len(cand[key])}")
            for k, v in list(cand[key].items())[:10]:
                print(f"        {k}: {v}")

    b_ok = base["_counts"]["import_ok"]
    c_ok = cand["_counts"]["import_ok"]
    if c_ok < b_ok:
        rc = 1
        print(f"FAIL  [{tag}] import_ok fell {b_ok} -> {c_ok}: a module stopped being imported.")
        lost = set(k for k, v in base["_detail_import"].items() if v == "ok") - \
               set(k for k, v in cand["_detail_import"].items() if v == "ok")
        for k in sorted(lost)[:10]:
            print(f"        no longer imported (or renamed): {k}")
        print("        If this is a deliberate rename, confirm the new path imports, then re-baseline.")

    if rc == 0:
        c = cand["_counts"]
        print(f"OK    [{tag}] ast={c['ast_files']} import_ok={c_ok} (baseline {b_ok}) "
              f"skipped={c['import_skipped']} failed={c['import_failed']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
