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
    REPO / "proforma" / "ui",                        # current
    REPO / "proforma" / "v1_5" / "ui",               # tag proforma-v1.5
    REPO / "earnest-proforma-2.0" / "streamlits",    # tag pre-refactor
]
UI_DIR = next((d for d in _CANDIDATES if (d / "app.py").is_file()), _CANDIDATES[0])
SCRIPT = UI_DIR / "app.py"


def assert_no_auto_pages() -> None:
    """A directory literally named `pages/` beside the entrypoint silently becomes multipage nav.

    streamlit's `_mpa_v1` (runtime/scriptrunner/script_runner.py) globs `<entrypoint_dir>/pages/*.py`
    and turns EVERY match into a sidebar nav page. Its only exclusions are names starting with `.`
    and `__init__.py` -- a LEADING UNDERSCORE IS NOT SKIPPED in streamlit 1.58, contrary to a common
    belief carried over from older versions.

    So putting helper modules in `ui/pages/` adds phantom nav entries that execute those helpers as
    standalone scripts when clicked. That happened here once: three helper modules under `ui/pages/`
    turned a 1-page app into a 4-page one. AppTest renders only the default page, so the golden
    widget surface did NOT catch it. Hence this explicit check. The helpers live in `ui/panels/`.
    """
    from pathlib import Path as _P

    pages_dir = _P(SCRIPT).resolve().parent / "pages"
    if not pages_dir.exists():
        return
    stray = sorted(
        p.name for p in pages_dir.glob("*.py")
        if not p.name.startswith(".") and p.name != "__init__.py"
    )
    if stray:
        raise SystemExit(
            f"[capture_ui] FAIL: {len(stray)} file(s) in {pages_dir} would become streamlit nav "
            f"pages: {stray}\n  A leading underscore does NOT exempt them. Move them out of a "
            f"directory named 'pages'."
        )


def main(out_dir: str) -> None:
    assert_no_auto_pages()
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
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/_golden/baseline")
