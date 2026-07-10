"""Compare two golden JSON captures. Exit 0 iff every numeric leaf matches to TOL.

    python scripts/_golden/diff.py <baseline.json> <candidate.json> [tol]

Numeric comparison is absolute-or-relative (`abs(a-b) <= tol + tol*abs(b)`), so it is
meaningful for both wash counts (~1e4) and shares (~1e-1). NaN == NaN and
Inf == Inf are treated as equal: the model emits NaN for a suppressed anchor, and
that is a real, load-bearing value we want held constant.

Keys beginning with `_` are metadata (module origin, artifact path) and are reported
as informational drift, not failure -- the whole point of the refactor is that
`_origin` changes while the numbers do not.
"""
from __future__ import annotations

import json
import math
import sys

TOL = 1e-9
diffs: list[str] = []
info: list[str] = []


def num_eq(a: float, b: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= TOL + TOL * abs(b)


def walk(a, b, path: str) -> None:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(f"{path}.{k}: missing in baseline")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing in candidate")
            else:
                walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)):
        if not num_eq(float(a), float(b)):
            diffs.append(f"{path}: {a!r} != {b!r}  (delta {float(a) - float(b):.3e})")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    global TOL
    base_p, cand_p = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        TOL = float(sys.argv[3])
    with open(base_p) as f:
        base = json.load(f)
    with open(cand_p) as f:
        cand = json.load(f)

    for k in sorted(set(base) | set(cand)):
        if k.startswith("_"):
            if base.get(k) != cand.get(k):
                info.append(f"  (info) {k}: {base.get(k)!r} -> {cand.get(k)!r}")
            continue
        walk(base.get(k), cand.get(k), k)

    for line in info:
        print(line)
    if diffs:
        print(f"\nFAIL  {len(diffs)} numeric/structural difference(s) at tol={TOL:g}:")
        for d in diffs[:40]:
            print("  " + d)
        if len(diffs) > 40:
            print(f"  ... and {len(diffs) - 40} more")
        return 1
    print(f"OK    {base_p} == {cand_p}  (tol={TOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
