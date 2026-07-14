"""Pin-cache DB maintenance — run on the box that can reach the DB (e.g. prod 10.53.222.4).

    conda activate sonnys
    python -m app.db.maintenance create     # CREATE TABLE IF NOT EXISTS (idempotent; also runs on API boot)
    python -m app.db.maintenance stats       # row counts, per-endpoint rows/hits, hits served
    python -m app.db.maintenance prune        # delete rows past their TTL
    python -m app.db.maintenance clear [--endpoint pinpoint-forecast]   # wipe all (or one endpoint)
    python -m app.db.maintenance ping          # verify connectivity + show server version

`clear` (full wipe) asks for confirmation; pass --yes to skip. Use it after a model/panel retrain so
stale forecasts are recomputed on next request (TTL also handles this automatically over time)."""
from __future__ import annotations

import argparse
import json
import sys

from app.core import common as calib
from app.db import pin_cache
from app.db.engine import get_engine


def _require_engine():
    eng = get_engine()
    if eng is None:
        print("No DB engine: CAR_WASH_DB_URL is unset or the database is unreachable.", file=sys.stderr)
        raise SystemExit(2)
    return eng


def cmd_ping(_args) -> None:
    eng = _require_engine()
    from sqlalchemy import text
    with eng.connect() as conn:
        ver = conn.execute(text("SELECT VERSION()")).scalar()
    print(f"OK  connected to {eng.url.render_as_string(hide_password=True)}")
    print(f"    server version: {ver}")


def cmd_create(_args) -> None:
    ok = pin_cache.init(create=True, prune=False)
    print(f"table '{calib.PIN_CACHE_TABLE}' ready" if ok else "create failed (see logs)")
    if not ok:
        raise SystemExit(1)


def cmd_stats(_args) -> None:
    _require_engine()
    print(json.dumps(pin_cache.stats(), indent=2, default=str))


def cmd_prune(_args) -> None:
    _require_engine()
    n = pin_cache.prune_expired()
    print(f"pruned {n} expired row(s)")


def cmd_clear(args) -> None:
    _require_engine()
    scope = f"endpoint '{args.endpoint}'" if args.endpoint else "ALL endpoints"
    if not args.yes:
        reply = input(f"Delete cached rows for {scope}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return
    n = pin_cache.clear(endpoint=args.endpoint)
    print(f"cleared {n} row(s) for {scope}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m app.db.maintenance", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping", help="verify connectivity")
    sub.add_parser("create", help="create the cache table if absent")
    sub.add_parser("stats", help="show cache statistics")
    sub.add_parser("prune", help="delete expired rows")
    c = sub.add_parser("clear", help="delete all rows (or one endpoint's)")
    c.add_argument("--endpoint", default=None, help="only clear this endpoint (e.g. pinpoint-forecast)")
    c.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    args = p.parse_args(argv)
    {
        "ping": cmd_ping,
        "create": cmd_create,
        "stats": cmd_stats,
        "prune": cmd_prune,
        "clear": cmd_clear,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
