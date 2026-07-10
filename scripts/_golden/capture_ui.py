"""Capture the Streamlit app's RENDERED surface. Run in the conda `proforma311` env.

    python scripts/_golden/capture_ui.py <out_dir>

This closes the gap that `ast.parse` leaves. streamlit.testing.v1.AppTest actually EXECUTES the
script body -- imports, data loads, model load, widget construction -- and surfaces any exception,
so it catches the class of break that a syntax check cannot: a bad import, a wrong data path, a
constant that moved, a page module that fails to dispatch.

Two things it does not do, stated plainly rather than implied away:
  * It renders only the FIRST pass. Widgets that appear after a user picks a mode, drops a pin, or
    clicks a button are not exercised. Real interaction coverage would need per-mode AppTest runs.
  * It does not compare pixels. Layout and styling regressions are invisible to it.

AppTest does not run streamlit's bootstrap, so it does not put the script's own directory on
sys.path. `streamlit run` does (streamlit/web/bootstrap.py:59), and app.py relies on that to
`import site_visual_page`. We replicate exactly that one insert -- nothing more -- so the harness
tests the app as `streamlit run` would load it, not some easier variant.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Post-refactor location first; the pre-refactor one so this same harness can be pointed at the
# `pre-refactor` tag in a worktree and prove the golden is a pre-refactor artifact, not a
# post-refactor one blessed after the fact.
_CANDIDATES = [
    REPO / "proforma" / "v1_5" / "ui",
    REPO / "earnest-proforma-2.0" / "streamlits",
]
UI_DIR = next((d for d in _CANDIDATES if (d / "app.py").is_file()), _CANDIDATES[0])
SCRIPT = UI_DIR / "app.py"


def main(out_dir: str) -> None:
    sys.path.insert(0, str(UI_DIR))  # exactly what streamlit run does; see docstring
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(SCRIPT), default_timeout=900)
    at.run()

    if at.exception:
        msgs = [str(e.value) for e in at.exception]
        print("[capture_ui] SCRIPT RAISED:")
        for m in msgs:
            print("   ", m)
        raise SystemExit(1)

    out = {
        "_script": str(SCRIPT.relative_to(REPO)),
        "counts": {
            "button": len(at.button),
            "checkbox": len(at.checkbox),
            "error": len(at.error),
            "markdown": len(at.markdown),
            "radio": len(at.radio),
            "selectbox": len(at.selectbox),
            "sidebar_radio": len(at.sidebar.radio),
            "sidebar_selectbox": len(at.sidebar.selectbox),
            "slider": len(at.slider),
            "tabs": len(at.tabs),
            "text_input": len(at.text_input),
        },
        # first markdown block is the injected theme CSS -- a cheap fingerprint of the page head
        "markdown_head": (at.markdown[0].value[:200] if len(at.markdown) else None),
        "button_labels": sorted(b.label for b in at.button),
        "text_input_labels": sorted(t.label for t in at.text_input),
    }

    dest = Path(out_dir) / "ui_render.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"[capture_ui] app body executed cleanly, 0 exceptions -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/_refactor/baseline")
