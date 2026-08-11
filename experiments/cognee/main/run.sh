#!/usr/bin/env bash
# Cognee demo launcher. Uses the isolated venv — never touches conda `sonnys`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/../.venv-cognee"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating isolated venv at $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q cognee
fi

# Unbuffered: the demo narrates as it works. Buffered stdout makes a live run
# look frozen for minutes and then dump everything at once.
export PYTHONUNBUFFERED=1
exec "$VENV/bin/python" "$HERE/demo.py" "$@"
