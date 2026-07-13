"""
The Council Chamber — a self-contained, animated boardroom for the site-selection committee.

`build_chamber_html(data)` returns ONE complete `<!doctype html>` document (a string). The Streamlit
view embeds it verbatim with `st.components.v1.html(...)`. This module therefore imports NOTHING from
streamlit / app.* / proforma.* — only `experiments.council.config` (for colour fallbacks) and stdlib.

What it draws, from the `data` dict (shape documented in the task / README):
  • a verdict banner (label · confidence · P(good build) · basis · optional divergence warning);
  • a numbers strip of stat tiles (revenue, net, breakeven, membership, washes, tunnel, CAPEX);
  • a round conference table with the 5 expert figurines seated around it, the facilitator at the head,
    and the leakage-clean data-signal "exhibit" laid on the table;
  • an animated playback of `messages`: a speech bubble pops over the speaker (coloured by message type),
    an arrow flies from challenger → target, confidence meters pulse/update from `belief_history`, and
    ballots appear on the VOTE round — driven by a ▶/⏸/⟲ transport with a scrub slider and speed control.

Everything is inline (SVG figurines, CSS keyframes, vanilla JS). No CDN, no external font/image — a strict
component CSP would block them. The `data` dict is injected as an html-safe JSON blob and vanilla JS builds
and animates the DOM. Run `python -m experiments.council.chamber` to print a demo document.
"""
from __future__ import annotations

import html as _html  # noqa: F401  (kept available; text is rendered via textContent, so this stays a guard)
import json

from experiments.council import config as C


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────
def build_chamber_html(data: dict, *, height: int = 640) -> str:
    """Return a complete self-contained HTML document (string) rendering the animated council chamber.

    `data` follows the committee's chamber schema (verdict / experts / facilitator / signal / numbers /
    messages / belief_history). Missing keys degrade gracefully — the JS is defensive. `height` is the
    iframe height the caller will pass to `components.html`; it is used as a hard fallback for `100vh`.
    """
    data = data if isinstance(data, dict) else {}
    cfg = {
        "msg_colors": getattr(C, "MSG_COLORS", {}),
        "verdict_colors": getattr(C, "VERDICT_COLORS", {}),
        "expert_meta": getattr(C, "EXPERT_META", {}),
        "verdict_labels": getattr(C, "VERDICT_LABELS", {}),
    }
    try:
        h = int(height)
    except Exception:
        h = 640
    h = max(360, min(2400, h))
    return (
        _TEMPLATE
        .replace("__DATA_JSON__", _safe_json(data))
        .replace("__CFG_JSON__", _safe_json(cfg))
        .replace("__HEIGHT__", str(h))
    )


def _safe_json(obj) -> str:
    """json.dumps that is safe to drop into a `<script>`: the only chars that could close the script
    or be reparsed as HTML (`<` `>` `&`) and the two JS line separators live exclusively inside JSON
    string values, so escaping them to `\\uXXXX` keeps the blob a valid JS object literal."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return (
        s.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


# ─────────────────────────────────────────────────────────────────────────────
# the document template  (sentinels: __DATA_JSON__ · __CFG_JSON__ · __HEIGHT__)
# raw string so backticks / ${} in the embedded JS survive; no triple-double-quote inside.
# ─────────────────────────────────────────────────────────────────────────────
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Council Chamber</title>
<style>
  :root{
    --vh: __HEIGHT__px;
    /* boardroom neutrals — light */
    --plane:#eef1f5; --surface:#ffffff; --surface-2:#f7f9fc;
    --ink:#0f172a; --ink-2:#475569; --muted:#94a3b8;
    --border:rgba(15,23,42,.10); --rim:rgba(15,23,42,.16); --shadow:rgba(15,23,42,.16);
    --table-a:#eaf0f7; --table-b:#d3ddec; --table-rim:rgba(15,23,42,.10); --table-glow:rgba(255,255,255,.75);
    --chip:#f1f5f9; --track:rgba(15,23,42,.08);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --plane:#0b1020; --surface:#141a2b; --surface-2:#1b2233;
      --ink:#f1f5f9; --ink-2:#cbd5e1; --muted:#8593a8;
      --border:rgba(255,255,255,.12); --rim:rgba(255,255,255,.16); --shadow:rgba(0,0,0,.55);
      --table-a:#1e2942; --table-b:#131b2e; --table-rim:rgba(255,255,255,.10); --table-glow:rgba(120,150,220,.18);
      --chip:#20293c; --track:rgba(255,255,255,.10);
    }
  }
  /* explicit toggle overrides the media query (higher specificity, wins both ways) */
  :root[data-theme="light"]{
    --plane:#eef1f5; --surface:#ffffff; --surface-2:#f7f9fc;
    --ink:#0f172a; --ink-2:#475569; --muted:#94a3b8;
    --border:rgba(15,23,42,.10); --rim:rgba(15,23,42,.16); --shadow:rgba(15,23,42,.16);
    --table-a:#eaf0f7; --table-b:#d3ddec; --table-rim:rgba(15,23,42,.10); --table-glow:rgba(255,255,255,.75);
    --chip:#f1f5f9; --track:rgba(15,23,42,.08);
  }
  :root[data-theme="dark"]{
    --plane:#0b1020; --surface:#141a2b; --surface-2:#1b2233;
    --ink:#f1f5f9; --ink-2:#cbd5e1; --muted:#8593a8;
    --border:rgba(255,255,255,.12); --rim:rgba(255,255,255,.16); --shadow:rgba(0,0,0,.55);
    --table-a:#1e2942; --table-b:#131b2e; --table-rim:rgba(255,255,255,.10); --table-glow:rgba(120,150,220,.18);
    --chip:#20293c; --track:rgba(255,255,255,.10);
  }

  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink); background:var(--plane);
    -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  }
  .wrap{
    height:var(--vh); min-height:var(--vh);
    display:flex; flex-direction:column; gap:10px; padding:12px;
    overflow-y:auto; overflow-x:hidden;
    background:
      radial-gradient(120% 80% at 50% -10%, color-mix(in srgb, var(--surface) 55%, transparent), transparent 60%),
      var(--plane);
  }

  /* ── verdict banner ── */
  .banner{
    position:relative; flex:0 0 auto; display:flex; flex-wrap:wrap; align-items:center; gap:14px;
    padding:12px 14px 12px 18px; border-radius:14px;
    background:var(--surface); border:1px solid var(--border);
    box-shadow:0 1px 2px var(--shadow); overflow:hidden;
  }
  .banner::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--vc,#64748b)}
  .banner .vwrap{display:flex;align-items:center;gap:12px;min-width:210px}
  .gavel{
    width:44px;height:44px;flex:0 0 auto;border-radius:12px;display:grid;place-items:center;font-size:24px;
    background:color-mix(in srgb, var(--vc) 16%, var(--surface)); border:1px solid color-mix(in srgb, var(--vc) 40%, transparent);
  }
  .vhead{display:flex;flex-direction:column;line-height:1.12}
  .vkick{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
  .vlabel{font-size:23px;font-weight:800;color:var(--vc);letter-spacing:-.01em}
  .vbasis{font-size:11.5px;color:var(--ink-2);margin-top:1px;max-width:340px}
  .vspacer{flex:1 1 20px}
  .vstats{display:flex;gap:8px;flex-wrap:wrap;align-items:stretch}
  .vstat{
    min-width:92px;padding:7px 12px;border-radius:11px;background:var(--surface-2);border:1px solid var(--border);
    display:flex;flex-direction:column;gap:1px;
  }
  .vstat .k{font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
  .vstat .v{font-size:18px;font-weight:800;color:var(--ink)}
  .vmeta{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .pill{
    font-size:10.5px;font-weight:600;color:var(--ink-2);background:var(--surface-2);
    border:1px solid var(--border);border-radius:999px;padding:3px 9px;white-space:nowrap;
  }
  .note{
    flex:1 1 100%;margin-top:2px;display:flex;gap:8px;align-items:flex-start;
    padding:8px 11px;border-radius:10px;font-size:12px;line-height:1.35;color:var(--ink);
    background:color-mix(in srgb,#f59e0b 15%, var(--surface));
    border:1px solid color-mix(in srgb,#f59e0b 45%, transparent);
  }
  .note b{white-space:nowrap}

  /* ── numbers strip ── */
  .numbers{flex:0 0 auto;display:flex;gap:8px;overflow-x:auto;overflow-y:hidden;padding-bottom:2px;scrollbar-width:thin}
  .tile{
    flex:1 1 auto;min-width:96px;padding:8px 11px;border-radius:11px;
    background:var(--surface);border:1px solid var(--border);box-shadow:0 1px 2px var(--shadow);
    display:flex;flex-direction:column;gap:2px;
  }
  .tile .k{font-size:10px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
  .tile .v{font-size:17px;font-weight:700;color:var(--ink);line-height:1.05}
  .tile .s{font-size:10.5px;color:var(--ink-2)}

  /* ── stage / boardroom ── */
  .stage-wrap{position:relative;flex:1 1 auto;min-height:356px}
  .stage{
    position:absolute;inset:0;border-radius:16px;overflow:hidden;
    background:
      radial-gradient(75% 62% at 50% 46%, color-mix(in srgb,var(--surface) 70%, transparent), transparent 72%),
      var(--surface-2);
    border:1px solid var(--border);box-shadow:inset 0 1px 0 var(--table-glow), 0 1px 2px var(--shadow);
  }
  .roomtag{
    position:absolute;top:9px;left:12px;z-index:6;display:flex;gap:7px;align-items:center;
    font-size:11px;color:var(--muted);font-weight:600;
  }
  .roomtag .dot{width:7px;height:7px;border-radius:50%;background:var(--vc,#64748b);box-shadow:0 0 0 3px color-mix(in srgb,var(--vc) 22%, transparent)}

  .table{
    position:absolute;left:50%;top:51%;transform:translate(-50%,-50%);
    width:58%;height:47%;border-radius:50%;
    background:radial-gradient(120% 120% at 50% 22%, var(--table-a), var(--table-b));
    border:1px solid var(--table-rim);
    box-shadow:0 26px 46px -22px var(--shadow), inset 0 2px 6px var(--table-glow), inset 0 -18px 34px -20px rgba(0,0,0,.28);
  }
  .table::before{
    content:"";position:absolute;left:50%;top:52%;transform:translate(-50%,-50%);
    width:78%;height:70%;border-radius:50%;border:1px dashed var(--table-rim);opacity:.6;
  }

  /* signal exhibit on the table */
  .exhibit{
    position:absolute;left:50%;top:51%;transform:translate(-50%,-50%) rotate(-3.5deg);
    width:min(46%,158px);padding:9px 11px;border-radius:11px;z-index:2;
    background:var(--surface);border:1px solid var(--border);
    box-shadow:0 12px 22px -12px var(--shadow), 0 1px 0 var(--table-glow);
  }
  .exhibit .eh{display:flex;align-items:center;gap:6px;font-size:10px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
  .exhibit .er{display:flex;align-items:baseline;gap:7px;margin-top:4px;flex-wrap:wrap}
  .exhibit .lean{
    font-size:12.5px;font-weight:800;color:#fff;padding:2px 9px;border-radius:999px;white-space:nowrap;
    background:var(--sc,#0891b2);box-shadow:0 1px 2px var(--shadow);
  }
  .exhibit .prob{font-size:12px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums}
  .exhibit .sub{font-size:10px;color:var(--muted);margin-top:3px}
  .exhibit::after{ /* a paper-clip glint */
    content:"";position:absolute;top:-5px;right:14px;width:10px;height:22px;border-radius:6px;
    border:2px solid var(--muted);opacity:.5;transform:rotate(12deg);
  }

  svg.wires{position:absolute;inset:0;width:100%;height:100%;z-index:3;pointer-events:none;overflow:visible}
  .wpath{fill:none;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;
    stroke-dasharray:var(--len);stroke-dashoffset:var(--len);opacity:.95;
    transition:stroke-dashoffset .5s ease-out}
  .wpath.on{stroke-dashoffset:0}
  .whead{transition:opacity .2s ease .32s;opacity:0}
  .whead.on{opacity:1}

  .seats{position:absolute;inset:0;z-index:4}
  .seat{
    position:absolute;transform:translate(-50%,-50%);width:130px;
    display:flex;flex-direction:column;align-items:center;gap:3px;
    opacity:0;animation:seatIn .5s ease forwards;
  }
  @keyframes seatIn{from{opacity:0;transform:translate(-50%,-40%) scale(.9)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}
  .fig{position:relative;width:74px;height:80px}
  .halo{
    position:absolute;left:50%;top:26px;transform:translate(-50%,-50%);
    width:62px;height:62px;border-radius:50%;border:3px solid var(--lean,#94a3b8);
    box-shadow:0 0 0 0 transparent;transition:border-color .35s ease,box-shadow .35s ease;z-index:0;
  }
  .fig.lit .halo{box-shadow:0 0 16px 1px color-mix(in srgb,var(--lean) 60%, transparent)}
  .bust{position:absolute;left:50%;bottom:0;transform:translateX(-50%);width:74px;height:42px;z-index:1;
    filter:drop-shadow(0 6px 8px color-mix(in srgb,var(--c) 30%, transparent))}
  .head{
    position:absolute;left:50%;top:0;transform:translateX(-50%);
    width:52px;height:52px;border-radius:50%;display:grid;place-items:center;z-index:2;
    background:radial-gradient(120% 120% at 50% 22%, color-mix(in srgb,var(--c) 22%, var(--surface)), color-mix(in srgb,var(--c) 34%, var(--surface)));
    border:2.5px solid var(--c);box-shadow:0 4px 10px -4px var(--shadow), inset 0 2px 4px rgba(255,255,255,.35);
  }
  .head .face{font-size:26px;line-height:1;filter:saturate(1.05)}
  .speaking .head{animation:bob 1.2s ease-in-out infinite}
  @keyframes bob{0%,100%{transform:translateX(-50%) translateY(0)}50%{transform:translateX(-50%) translateY(-3px)}}
  .ballot{
    position:absolute;top:-2px;right:2px;z-index:5;width:23px;height:23px;border-radius:50%;
    display:grid;place-items:center;font-size:13px;font-weight:900;color:#fff;
    background:var(--bc,#64748b);border:2px solid var(--surface);box-shadow:0 2px 5px var(--shadow);
    transform:scale(0);transition:transform .3s cubic-bezier(.2,1.5,.4,1)}
  .ballot.show{transform:scale(1)}

  .plate{
    font-size:11.5px;font-weight:700;color:var(--ink);background:var(--surface);
    border:1px solid var(--border);border-radius:999px;padding:2px 9px;white-space:nowrap;
    box-shadow:0 1px 2px var(--shadow);
  }
  .seat.facil .plate{background:var(--ink);color:var(--surface);border-color:transparent}
  .subrole{font-size:9.5px;color:var(--muted);margin-top:-1px}
  .meter{width:98px;height:8px;border-radius:999px;background:var(--track);overflow:hidden;position:relative}
  .mfill{height:100%;width:0;border-radius:999px;background:var(--c,#64748b);transition:width .6s cubic-bezier(.3,.8,.3,1)}
  .meter.pulse{animation:mpulse .9s ease}
  @keyframes mpulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--c) 60%, transparent)}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0 transparent}}
  .keynum{font-size:10px;color:var(--ink-2);max-width:126px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .keynum b{font-weight:800;color:var(--ink);font-variant-numeric:tabular-nums}
  .keynum span{color:var(--muted)}

  /* speech bubble (one per seat; only the active one shows) */
  .bubble{
    position:absolute;left:50%;bottom:calc(100% + 10px);transform:translate(-50%,6px) scale(.9);
    width:max-content;max-width:198px;z-index:30;
    background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--tc,#64748b);
    border-radius:12px;padding:7px 9px;box-shadow:0 14px 30px -10px var(--shadow), 0 2px 6px var(--shadow);
    opacity:0;pointer-events:none;transition:opacity .28s ease, transform .28s cubic-bezier(.2,1.3,.4,1);
  }
  .seat.flip .bubble{bottom:auto;top:calc(100% + 10px);transform:translate(-50%,-6px) scale(.9)}
  .bubble.show{opacity:1;transform:translate(-50%,0) scale(1)}
  .bubble::after{ /* tail */
    content:"";position:absolute;left:50%;top:100%;transform:translateX(-50%);
    border:7px solid transparent;border-top-color:var(--surface);filter:drop-shadow(0 1px 0 var(--border));
  }
  .seat.flip .bubble::after{top:auto;bottom:100%;border-top-color:transparent;border-bottom-color:var(--tc,#64748b)}
  .btag{display:inline-flex;align-items:center;gap:5px;font-size:9.5px;font-weight:800;letter-spacing:.07em;
    color:#fff;background:var(--tc,#64748b);border-radius:999px;padding:2px 8px;text-transform:uppercase}
  .btag .who{opacity:.85;font-weight:700;letter-spacing:.02em;text-transform:none}
  .btext{margin-top:6px;font-size:12px;line-height:1.32;color:var(--ink);display:flex;gap:6px}
  .btext .bemoji{font-size:15px;line-height:1.2;flex:0 0 auto}
  .bto{margin-top:5px;font-size:10px;color:var(--muted);display:flex;align-items:center;gap:4px}
  .bcites{margin-top:6px;display:flex;flex-wrap:wrap;gap:4px}
  .cite{font-size:9.5px;font-weight:600;color:var(--ink-2);background:var(--chip);border:1px solid var(--border);
    border-radius:6px;padding:1px 6px;font-variant-numeric:tabular-nums}

  /* ── controls ── */
  .controls{
    flex:0 0 auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    padding:9px 12px;border-radius:13px;background:var(--surface);border:1px solid var(--border);
    box-shadow:0 1px 2px var(--shadow);
  }
  .btn{
    display:inline-flex;align-items:center;gap:6px;font:inherit;font-size:13px;font-weight:700;color:var(--ink);
    background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:7px 12px;cursor:pointer;
    transition:filter .15s ease, transform .05s ease;
  }
  .btn:hover{filter:brightness(1.04)}
  .btn:active{transform:translateY(1px)}
  .btn.primary{color:#fff;background:var(--accent,#2563eb);border-color:transparent;min-width:96px;justify-content:center}
  .round{font-size:12px;font-weight:800;color:var(--ink);letter-spacing:.02em;min-width:118px}
  .round .rdot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent,#2563eb);margin-right:6px;vertical-align:middle}
  .slider{flex:1 1 160px;min-width:120px;accent-color:var(--accent,#2563eb);height:22px}
  .progress{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums;min-width:44px;text-align:right}
  .spd{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--ink-2)}
  .spd select{font:inherit;font-size:12px;color:var(--ink);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:4px 6px}
  .icobtn{width:34px;height:34px;padding:0;justify-content:center;font-size:15px}

  /* ── caption: an always-visible subtitle so a clipped speech bubble is still readable ── */
  .caption{flex:0 0 auto;margin-top:8px;padding:9px 12px;border-radius:10px;background:var(--surface);
    border:1px solid var(--border);color:var(--ink);font-size:13px;line-height:1.42;display:flex;gap:9px;
    align-items:flex-start;box-shadow:0 1px 2px var(--shadow);min-height:20px}
  .caption.empty{color:var(--ink-2);font-style:italic;justify-content:center;align-items:center}
  .caption .ctag{flex:0 0 auto;font-weight:800;font-size:10px;letter-spacing:.03em;padding:3px 7px;
    border-radius:6px;color:#fff;background:var(--cc,#64748b);white-space:nowrap}
  .caption .cwho{font-weight:700}
  /* ── fullscreen (native): fill the screen + give the stage room so bubbles never clip ── */
  .wrap:fullscreen, .wrap:-webkit-full-screen{height:100vh;min-height:100vh;width:100vw;padding:18px 20px;
    background:var(--plane);overflow-y:auto}
  .wrap:fullscreen .stage-wrap{min-height:56vh}

  @media (max-width:560px){
    .seat{width:112px}.fig{width:64px;height:72px}.head{width:46px;height:46px}.head .face{font-size:22px}
    .halo{width:56px;height:56px;top:23px}.bust{width:64px;height:38px}.meter{width:86px}
    .vlabel{font-size:20px}.tile{min-width:94px}
  }
  @media (prefers-reduced-motion: reduce){
    *{animation-duration:.001ms !important;transition-duration:.05s !important}
    .speaking .head{animation:none}
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="banner" id="banner"></header>
    <div class="numbers" id="numbers" aria-label="site economics"></div>
    <div class="stage-wrap">
      <div class="stage" id="stage">
        <div class="roomtag" id="roomtag"></div>
        <div class="table"><div class="table-inner"></div></div>
        <div class="exhibit" id="exhibit"></div>
        <svg class="wires" id="wires" preserveAspectRatio="none"></svg>
        <div class="seats" id="seats"></div>
      </div>
    </div>
    <div class="caption empty" id="caption" aria-live="polite">Press ▶ Play to watch the committee deliberate — each seat's message shows here too.</div>
    <div class="controls">
      <button class="btn primary" id="play">▶&nbsp;Play</button>
      <button class="btn icobtn" id="restart" title="Restart">⟲</button>
      <input class="slider" id="slider" type="range" min="0" max="0" value="0" step="1" aria-label="scrub discussion"/>
      <span class="round" id="round"></span>
      <span class="progress" id="progress"></span>
      <span class="spd">Speed
        <select id="speed" aria-label="playback speed">
          <option value="0.5">0.5&times;</option>
          <option value="1" selected>1&times;</option>
          <option value="1.5">1.5&times;</option>
          <option value="2">2&times;</option>
        </select>
      </span>
      <button class="btn icobtn" id="theme" title="Toggle light / dark">◐</button>
      <button class="btn icobtn" id="fs" title="Fullscreen">⛶</button>
    </div>
  </div>

  <script>window.__CHAMBER__ = __DATA_JSON__; window.__CFG__ = __CFG_JSON__;</script>
  <script>
  (function(){
    "use strict";
    var D   = window.__CHAMBER__ || {};
    var CFG = window.__CFG__ || {};
    var MSGC = CFG.msg_colors || {};
    var VC   = CFG.verdict_colors || {Build:"#16a34a",Pass:"#dc2626",Conditional:"#f59e0b",Insufficient:"#64748b"};

    var experts   = Array.isArray(D.experts) ? D.experts : [];
    var messages  = Array.isArray(D.messages) ? D.messages : [];
    var belief    = D.belief_history || {};
    var facil     = D.facilitator || {name:"Facilitator", emoji:"🧭", color:"#0f172a"};
    var signal    = D.signal || {};
    var numbers   = D.numbers || {};

    // ── tiny helpers ───────────────────────────────────────────────
    function $(id){ return document.getElementById(id); }
    function el(tag, cls, txt){ var e=document.createElement(tag); if(cls) e.className=cls; if(txt!=null) e.textContent=txt; return e; }
    function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
    function pct(x){ return Math.round((Number(x)||0)*100); }
    function num(x){ return (x==null || x==="") ? null : Number(x); }
    function hexA(hex, a){
      hex=String(hex||"#64748b").replace("#","");
      if(hex.length===3) hex=hex.split("").map(function(c){return c+c;}).join("");
      var n=parseInt(hex,16); if(!isFinite(n)) return "rgba(100,116,139,"+a+")";
      return "rgba("+((n>>16)&255)+","+((n>>8)&255)+","+(n&255)+","+a+")";
    }
    function intc(v){ v=Number(v); return isFinite(v)? Math.round(v).toLocaleString("en-US") : ""; }
    function money(v){
      v=Number(v); if(!isFinite(v)) return "";
      var s=v<0?"-":""; v=Math.abs(v); var o;
      if(v>=1e9) o=(v/1e9).toFixed(1).replace(/\.0$/,"")+"B";
      else if(v>=1e6) o=(v/1e6).toFixed(1).replace(/\.0$/,"")+"M";
      else if(v>=1e3) o=Math.round(v/1e3)+"K";
      else o=String(Math.round(v));
      return s+"$"+o;
    }
    function leanColor(lean, fallback){
      if(!lean) return fallback||"#94a3b8";
      return VC[lean] || fallback || "#94a3b8";
    }

    var expertByKey = {};
    experts.forEach(function(e){ if(e && e.key) expertByKey[e.key]=e; });

    // ── verdict banner ─────────────────────────────────────────────
    var vcol = D.verdict_color || VC[D.verdict] || "#64748b";
    var accent = (experts[0] && experts[0].color) || "#2563eb";
    document.documentElement.style.setProperty("--accent", accent);
    (function banner(){
      var b=$("banner"); b.style.setProperty("--vc", vcol);
      var vwrap=el("div","vwrap");
      vwrap.appendChild((function(){ var g=el("div","gavel"); g.textContent="⚖️"; return g; })());
      var head=el("div","vhead");
      head.appendChild(el("div","vkick","Committee verdict"));
      head.appendChild(el("div","vlabel", D.verdict_label || D.verdict || "—"));
      if(D.basis) head.appendChild(el("div","vbasis", D.basis));
      vwrap.appendChild(head); b.appendChild(vwrap);
      b.appendChild(el("div","vspacer"));

      var stats=el("div","vstats");
      function stat(k,v){ var s=el("div","vstat"); s.appendChild(el("div","k",k)); s.appendChild(el("div","v",v)); return s; }
      if(D.confidence!=null) stats.appendChild(stat("Committee confidence", pct(D.confidence)+"%"));
      if(D.prob!=null)       stats.appendChild(stat("Signal · P(good build)", pct(D.prob)+"%"));
      b.appendChild(stats);

      var meta=el("div","vmeta");
      if(D.rounds!=null) meta.appendChild(el("span","pill", D.rounds+" round"+(D.rounds==1?"":"s")));
      if(D.consensus_pct!=null) meta.appendChild(el("span","pill", pct(D.consensus_pct)+"% consensus"));
      if(D.open_challenges!=null) meta.appendChild(el("span","pill", D.open_challenges+" standing disagreement"+(D.open_challenges==1?"":"s")));
      if(D.site && D.site.lat!=null) meta.appendChild(el("span","pill","📍 "+Number(D.site.lat).toFixed(3)+", "+Number(D.site.lon).toFixed(3)));
      if(meta.childNodes.length){ b.appendChild(el("div","vspacer")); b.appendChild(meta); }

      if(D.note){
        var n=el("div","note");
        n.appendChild(el("b",null,"Heads up"));
        n.appendChild(el("span",null, D.note));
        b.appendChild(n);
      }
    })();

    // ── numbers strip ──────────────────────────────────────────────
    (function strip(){
      var wrap=$("numbers");
      var tiles=[
        ["5-yr revenue", money(numbers.revenue_5yr), null],
        ["5-yr net",     money(numbers.net_5yr), null],
        ["Breakeven",    numbers.breakeven_month!=null ? "Mo "+intc(numbers.breakeven_month) : "", numbers.breakeven_month!=null?"months to payback":null],
        ["Membership",   numbers.membership_share!=null ? pct(numbers.membership_share)+"%" : "", "of washes"],
        ["Mature washes",numbers.mature_washes!=null ? intc(numbers.mature_washes) : "", "per month"],
        ["Tunnel",       numbers.tunnel_ft!=null ? intc(numbers.tunnel_ft)+" ft" : "", null],
        ["CAPEX",        money(numbers.capex), "to build"]
      ];
      tiles.forEach(function(t){
        if(!t[1]) return;
        var d=el("div","tile");
        d.appendChild(el("div","k",t[0]));
        d.appendChild(el("div","v",t[1]));
        if(t[2]) d.appendChild(el("div","s",t[2]));
        wrap.appendChild(d);
      });
      if(!wrap.childNodes.length) wrap.style.display="none";
    })();

    // room tag
    (function(){
      var r=$("roomtag"); r.style.setProperty("--vc", vcol);
      r.appendChild(el("span","dot"));
      r.appendChild(el("span",null,"Council Chamber · live deliberation"));
    })();

    // ── signal exhibit ─────────────────────────────────────────────
    (function exhibit(){
      var ex=$("exhibit"); var sc=leanColor(signal.lean,"#0891b2"); ex.style.setProperty("--sc", sc);
      var h=el("div","eh"); h.appendChild(el("span",null,"🎯")); h.appendChild(el("span",null,"Data signal")); ex.appendChild(h);
      var row=el("div","er");
      row.appendChild(el("span","lean", signal.lean || "—"));
      if(signal.prob!=null) row.appendChild(el("span","prob","P "+pct(signal.prob)+"%"));
      ex.appendChild(row);
      if(signal.confidence!=null) ex.appendChild(el("div","sub","cross-check · "+pct(signal.confidence)+"% conf"));
    })();

    // ── seat geometry & figurines ──────────────────────────────────
    var RX=39, CY=51, RY=30;                        // seat ellipse (percent of stage)
    var EXP_ANGLES=[210,150,90,30,-30];             // 5 experts, CCW from upper-left
    var seatLayer=$("seats"), wires=$("wires");
    var seats={};                                   // key -> {node, ax, ay, bubble, mfill, meter, keynum, ballot, fig, head, halo}

    function bust(color){
      var ns="http://www.w3.org/2000/svg";
      var svg=document.createElementNS(ns,"svg");
      svg.setAttribute("class","bust"); svg.setAttribute("viewBox","0 0 74 42");
      var p=document.createElementNS(ns,"path");
      p.setAttribute("d","M4 42 C4 20 22 15 37 15 C52 15 70 20 70 42 Z");
      p.setAttribute("fill", color);
      svg.appendChild(p);
      var collar=document.createElementNS(ns,"path");
      collar.setAttribute("d","M28 17 L37 27 L46 17");
      collar.setAttribute("fill","none"); collar.setAttribute("stroke","rgba(255,255,255,.55)"); collar.setAttribute("stroke-width","2.4");
      collar.setAttribute("stroke-linecap","round"); collar.setAttribute("stroke-linejoin","round");
      svg.appendChild(collar);
      return svg;
    }

    function makeSeat(key, meta, angleDeg, isFacil){
      var th=angleDeg*Math.PI/180;
      var ax=50 + RX*Math.cos(th);
      var ay=CY + RY*Math.sin(th);
      var color=meta.color || "#64748b";
      var node=el("div","seat"+(isFacil?" facil":""));
      node.style.left=ax+"%"; node.style.top=ay+"%"; node.style.setProperty("--c", color);
      if(ay<50) node.classList.add("flip");

      var fig=el("div","fig");
      var halo=el("div","halo"); halo.style.setProperty("--lean","#94a3b8");
      fig.appendChild(halo);
      fig.appendChild(bust(color));
      var head=el("div","head");
      head.appendChild(el("span","face", meta.emoji || "•"));
      fig.appendChild(head);
      var ballot=el("div","ballot");
      fig.appendChild(ballot);

      // bubble
      var bub=el("div","bubble");
      node.appendChild(fig);
      node.appendChild(bub);
      node.appendChild(el("div","plate", meta.name || key));

      var mfill=null, meter=null, keynum=null;
      if(isFacil){
        node.appendChild(el("div","subrole","chairs the room"));
      } else {
        meter=el("div","meter"); mfill=el("div","mfill");
        meter.style.background=hexA(color,0.16);
        meter.appendChild(mfill); node.appendChild(meter);
        keynum=el("div","keynum"); node.appendChild(keynum);
      }

      seatLayer.appendChild(node);
      seats[key]={node:node, ax:ax, ay:ay, bubble:bub, mfill:mfill, meter:meter, keynum:keynum, ballot:ballot, fig:fig, head:head, halo:halo, color:color};
    }

    // facilitator at the head
    makeSeat("facilitator", facil, -90, true);
    // experts around the table (in given order)
    experts.slice(0,5).forEach(function(e, i){
      makeSeat(e.key, {name:e.name, emoji:e.emoji, color:e.color}, EXP_ANGLES[i % EXP_ANGLES.length], false);
    });

    // ── belief lookup ──────────────────────────────────────────────
    function beliefAt(key, round){
      var hist=belief[key], exp=expertByKey[key]||{}, pick=null;
      if(Array.isArray(hist)){
        hist.forEach(function(h){
          var r=(h.round==null?0:h.round);
          if(r<=round && (pick===null || r>=(pick.round==null?0:pick.round))) pick=h;
        });
        if(pick===null && hist.length) pick=hist[0];
      }
      return {
        confidence: (pick && pick.confidence!=null) ? pick.confidence : (exp.confidence!=null?exp.confidence:0.5),
        key_number: (pick && pick.key_number!=null) ? pick.key_number : (exp.key_number!=null?exp.key_number:null),
        lean:       (pick && pick.lean) ? pick.lean : (exp.lean || null)
      };
    }
    function fmtKey(n, label){
      if(n==null) return {v:"", l:label||""};
      var L=String(label||"").toLowerCase(), v;
      if(/\$|capex|net|revenue|cost|price/.test(L)) v=money(n);
      else if((/%|share|rate|margin/.test(L)) && Math.abs(n)<=1) v=pct(n)+"%";
      else v=intc(n);
      return {v:v, l:label||""};
    }
    function applyBelief(key, round, pulse){
      var s=seats[key]; if(!s || !s.mfill) return;
      var exp=expertByKey[key]||{};
      var bs=beliefAt(key, round<0?0:round);
      s.mfill.style.width=clamp(pct(bs.confidence),0,100)+"%";
      var lc=(exp.lean_color && (round<0 || !bs.lean)) ? exp.lean_color : leanColor(bs.lean, exp.lean_color);
      s.halo.style.setProperty("--lean", lc);
      s.fig.classList.toggle("lit", !!bs.lean);
      if(s.keynum){
        var f=fmtKey(bs.key_number, exp.key_number_label);
        s.keynum.textContent="";
        if(f.v){ var b=el("b",null,f.v); s.keynum.appendChild(b); if(f.l){ s.keynum.appendChild(document.createTextNode(" ")); s.keynum.appendChild(el("span",null,f.l)); } }
      }
      if(pulse && s.meter){ s.meter.classList.remove("pulse"); void s.meter.offsetWidth; s.meter.classList.add("pulse"); }
    }

    // ── wires (arrows) ─────────────────────────────────────────────
    var lastWire=null;
    function clearWire(){ while(wires.firstChild) wires.removeChild(wires.firstChild); lastWire=null; }
    function drawWire(fromKey, toKey, color){
      var a=seats[fromKey], b=seats[toKey]; if(!a||!b) return;
      var sw=wires.clientWidth||1, sh=wires.clientHeight||1;
      wires.setAttribute("viewBox","0 0 "+sw+" "+sh);
      var x1=a.ax/100*sw, y1=a.ay/100*sh, x2=b.ax/100*sw, y2=b.ay/100*sh;
      var dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy)||1, ux=dx/len, uy=dy/len;
      var pad=34; x1+=ux*pad; y1+=uy*pad; x2-=ux*(pad+8); y2-=uy*(pad+8);
      var mx=(x1+x2)/2, my=(y1+y2)/2, curve=clamp(len*0.16, 12, 54);
      var cx=mx + (-uy)*curve, cy=my + (ux)*curve;                 // perpendicular bow
      var ns="http://www.w3.org/2000/svg";
      var path=document.createElementNS(ns,"path");
      path.setAttribute("d","M "+x1+" "+y1+" Q "+cx+" "+cy+" "+x2+" "+y2);
      path.setAttribute("class","wpath"); path.setAttribute("stroke",color);
      wires.appendChild(path);
      var L=path.getTotalLength(); path.style.setProperty("--len", L);
      // arrowhead — direction from control point to end
      var ang=Math.atan2(y2-cy, x2-cx), ah=9;
      var hx=Math.cos(ang), hy=Math.sin(ang), px=-hy, py=hx;
      var tip=document.createElementNS(ns,"polygon");
      var p1=(x2)+","+(y2);
      var p2=(x2-hx*ah + px*ah*0.62)+","+(y2-hy*ah + py*ah*0.62);
      var p3=(x2-hx*ah - px*ah*0.62)+","+(y2-hy*ah - py*ah*0.62);
      tip.setAttribute("points", p1+" "+p2+" "+p3);
      tip.setAttribute("fill", color); tip.setAttribute("class","whead");
      wires.appendChild(tip);
      // trigger draw-on
      void path.getBoundingClientRect(); path.classList.add("on"); tip.classList.add("on");
      lastWire={from:fromKey, to:toKey, color:color};
    }

    // ── bubbles & ballots ──────────────────────────────────────────
    function clearBubbles(){ Object.keys(seats).forEach(function(k){ seats[k].bubble.classList.remove("show"); seats[k].node.classList.remove("speaking"); }); }
    function fillBubble(seat, msg){
      var b=seat.bubble; b.textContent=""; var tc=msg.type_color || MSGC[msg.type] || "#64748b";
      b.style.setProperty("--tc", tc);
      var tag=el("div","btag"); tag.appendChild(el("span",null, msg.type||"MSG"));
      tag.appendChild(el("span","who", msg.sender_name || (expertByKey[msg.sender]&&expertByKey[msg.sender].name) || msg.sender || ""));
      b.appendChild(tag);
      var tx=el("div","btext");
      var em=el("span","bemoji", msg.sender_emoji || (expertByKey[msg.sender]&&expertByKey[msg.sender].emoji) || "");
      tx.appendChild(em);
      tx.appendChild(el("span",null, msg.text || ""));
      b.appendChild(tx);
      if(msg.to && (msg.to_name || seats[msg.to])){
        var tn=msg.to_name || (expertByKey[msg.to]&&expertByKey[msg.to].name) || msg.to;
        b.appendChild(el("div","bto","→ to "+tn));
      }
      if(Array.isArray(msg.cites) && msg.cites.length){
        var cw=el("div","bcites");
        msg.cites.slice(0,3).forEach(function(c){ cw.appendChild(el("span","cite", c)); });
        b.appendChild(cw);
      }
      void b.offsetWidth; b.classList.add("show"); seat.node.classList.add("speaking");
    }
    function ballotFor(key, curIdx){
      // reveal a ballot on any seat that has cast a VOTE at or before curIdx
      for(var i=0;i<=curIdx;i++){
        var m=messages[i]; if(!m || m.type!=="VOTE" || m.sender!==key) continue;
        var exp=expertByKey[key]||{}; var lean=exp.lean;
        var s=seats[key]; if(!s) return;
        var sym = lean==="Build"?"✓" : lean==="Pass"?"✕" : lean==="Conditional"?"~" : "•";
        s.ballot.textContent=sym; s.ballot.style.setProperty("--bc", exp.lean_color || leanColor(lean,"#64748b"));
        s.ballot.classList.add("show"); return;
      }
    }
    function updateBallots(curIdx){
      Object.keys(seats).forEach(function(k){ seats[k].ballot.classList.remove("show"); });
      Object.keys(seats).forEach(function(k){ ballotFor(k, curIdx); });
    }

    // ── round chrome ───────────────────────────────────────────────
    function roundText(curIdx){
      if(curIdx<0) return "Ready to convene";
      var m=messages[curIdx];
      if(m.type==="VOTE") return "Final vote";
      var r=m.round==null?0:m.round;
      if(r===0) return "Opening findings";
      return "Round "+r;
    }

    // ── timeline state machine ─────────────────────────────────────
    var step=0, playing=false, timer=null, speed=1, BASE=1600;
    var slider=$("slider"), playBtn=$("play"), roundEl=$("round"), progEl=$("progress");
    slider.max=messages.length;

    function setStep(s){
      step=clamp(s, 0, messages.length);
      slider.value=step;
      var curIdx=step-1;
      var round = curIdx>=0 ? (messages[curIdx].round==null?0:messages[curIdx].round) : -1;

      clearBubbles(); clearWire();
      // update every expert's meter/lean to the current round
      Object.keys(seats).forEach(function(k){ if(k!=="facilitator") applyBelief(k, round, false); });
      updateBallots(curIdx);

      if(curIdx>=0){
        var m=messages[curIdx];
        var sk=m.sender, seat=seats[sk] || seats["facilitator"];
        if(m.type==="REVISE") applyBelief(sk, round, true);
        if(seat) fillBubble(seat, m);
        if(m.to && seats[m.to] && seats[sk]) drawWire(sk, m.to, m.type_color || MSGC[m.type] || "#64748b");
      }

      // always-visible caption (subtitle) — readable even if the popup bubble is off-screen / clipped
      var cap=$("caption");
      if(curIdx>=0){
        var cm=messages[curIdx], cc=cm.type_color || MSGC[cm.type] || "#64748b";
        var cwho=(cm.sender_name||cm.sender||"")+(cm.to?(" → "+(cm.to_name||cm.to)):"");
        cap.className="caption"; cap.style.setProperty("--cc", cc); cap.textContent="";
        cap.appendChild(el("span","ctag", cm.type||"MSG"));
        var cs=el("span"); cs.appendChild(el("span","cwho", cwho+": ")); cs.appendChild(document.createTextNode(cm.text||"")); cap.appendChild(cs);
      } else {
        cap.className="caption empty"; cap.textContent="Press ▶ Play to watch the committee deliberate — each message shows here too.";
      }

      roundEl.textContent=""; roundEl.appendChild(el("span","rdot")); roundEl.appendChild(document.createTextNode(roundText(curIdx)));
      progEl.textContent = step + " / " + messages.length;
      $("stage").classList.toggle("done", step>=messages.length && messages.length>0);
    }

    function updatePlayBtn(){ playBtn.innerHTML = playing ? "⏸&nbsp;Pause" : (step>=messages.length ? "▶&nbsp;Replay" : "▶&nbsp;Play"); }
    function schedule(){ if(timer) clearInterval(timer); timer=setInterval(function(){
        if(step>=messages.length){ stop(); return; }
        setStep(step+1);
        if(step>=messages.length){ stop(); }
      }, BASE/speed); }
    function play(){
      if(!messages.length) return;
      if(step>=messages.length) setStep(0);
      playing=true; updatePlayBtn();
      if(step===0) setStep(1);
      schedule();
    }
    function stop(){ playing=false; if(timer){ clearInterval(timer); timer=null; } updatePlayBtn(); }
    function toggle(){ playing ? stop() : play(); }

    playBtn.onclick=toggle;
    $("restart").onclick=function(){ stop(); setStep(0); };
    slider.oninput=function(){ stop(); setStep(parseInt(slider.value,10)||0); };
    $("speed").onchange=function(){ speed=parseFloat(this.value)||1; if(playing) schedule(); };
    $("theme").onclick=function(){
      var r=document.documentElement, cur=r.getAttribute("data-theme");
      var dark = cur ? cur==="dark" : (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
      r.setAttribute("data-theme", dark ? "light" : "dark");
      if(lastWire) drawWire(lastWire.from, lastWire.to, lastWire.color);
    };
    $("fs").onclick=function(){
      var w=document.querySelector(".wrap");
      try{
        if(document.fullscreenElement||document.webkitFullscreenElement){ (document.exitFullscreen||document.webkitExitFullscreen).call(document); }
        else{ (w.requestFullscreen||w.webkitRequestFullscreen).call(w); }
      }catch(e){}
    };
    document.addEventListener("fullscreenchange", function(){
      var on=!!(document.fullscreenElement||document.webkitFullscreenElement);
      var fb=$("fs"); if(fb) fb.textContent = on ? "⤢" : "⛶";
      if(lastWire) setTimeout(function(){ drawWire(lastWire.from, lastWire.to, lastWire.color); }, 140);
    });

    // keep the active arrow correct across resizes
    var rz; window.addEventListener("resize", function(){
      clearTimeout(rz); rz=setTimeout(function(){ if(lastWire) drawWire(lastWire.from, lastWire.to, lastWire.color); }, 120);
    });

    // initial paint
    setStep(0);
    updatePlayBtn();
  })();
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# realistic demo payload (exercises every message type + belief evolution)
# ─────────────────────────────────────────────────────────────────────────────
DEMO_DATA = {
    "verdict": "Build",
    "verdict_label": "Build",
    "verdict_color": "#16a34a",
    "confidence": 0.78,
    "prob": 0.62,
    "basis": "committee consensus (data-weighted vote), signal concurring",
    "note": "⚠️ Competition flags 6 rivals within 3 mi — the mature anchor is trimmed ~6% but the lean holds.",
    "site": {"lat": 33.749, "lon": -84.388},
    "rounds": 3,
    "consensus_pct": 0.8,
    "open_challenges": 0,
    "experts": [
        {"key": "historical", "name": "Historical", "emoji": "\U0001F4CA", "color": "#2563eb",
         "lean": "Build", "lean_color": "#16a34a", "confidence": 0.70,
         "key_number": 11800.0, "key_number_label": "mature washes/mo"},
        {"key": "competition", "name": "Competition", "emoji": "\U0001F6F0️", "color": "#dc2626",
         "lean": "Conditional", "lean_color": "#f59e0b", "confidence": 0.58,
         "key_number": 6.0, "key_number_label": "rivals within 3 mi"},
        {"key": "local_market", "name": "Local-Market", "emoji": "\U0001F3D9️", "color": "#16a34a",
         "lean": "Build", "lean_color": "#16a34a", "confidence": 0.64,
         "key_number": 0.55, "key_number_label": "membership share"},
        {"key": "capacity", "name": "Capacity", "emoji": "\U0001F3D7️", "color": "#9333ea",
         "lean": "Build", "lean_color": "#16a34a", "confidence": 0.71,
         "key_number": 120.0, "key_number_label": "tunnel ft"},
        {"key": "finance", "name": "Finance", "emoji": "\U0001F4B0", "color": "#ea580c",
         "lean": "Build", "lean_color": "#16a34a", "confidence": 0.74,
         "key_number": 3100000.0, "key_number_label": "5-yr net $"},
    ],
    "facilitator": {"name": "Facilitator", "emoji": "\U0001F9ED", "color": "#0f172a"},
    "signal": {"lean": "Build", "prob": 0.62, "confidence": 0.66},
    "numbers": {"revenue_5yr": 22800000, "net_5yr": 3100000, "breakeven_month": 31,
                "membership_share": 0.55, "mature_washes": 12588, "tunnel_ft": 120, "capex": 1500000},
    "messages": [
        {"mid": "a1", "type": "PUBLISH", "type_color": "#64748b", "round": 0,
         "sender": "historical", "sender_name": "Historical", "sender_emoji": "\U0001F4CA", "sender_color": "#2563eb",
         "to": None, "to_name": None,
         "text": "The 12-mile neighbour cluster matures at ~12.6k washes/mo.", "cites": ["hist.cluster_wash"]},
        {"mid": "a2", "type": "PUBLISH", "type_color": "#64748b", "round": 0,
         "sender": "competition", "sender_name": "Competition", "sender_emoji": "\U0001F6F0️", "sender_color": "#dc2626",
         "to": None, "to_name": None,
         "text": "Google Places shows 6 express rivals within 3 mi, 9 within 5.", "cites": ["comp.count_3mi", "comp.count_5mi"]},
        {"mid": "a3", "type": "PUBLISH", "type_color": "#64748b", "round": 0,
         "sender": "capacity", "sender_name": "Capacity", "sender_emoji": "\U0001F3D7️", "sender_color": "#9333ea",
         "to": None, "to_name": None,
         "text": "Peak-hour throughput supports a 120 ft tunnel (100 ft + 20 buffer).", "cites": ["cap.tunnel_ft"]},
        {"mid": "b1", "type": "CHALLENGE", "type_color": "#ef4444", "round": 1,
         "sender": "competition", "sender_name": "Competition", "sender_emoji": "\U0001F6F0️", "sender_color": "#dc2626",
         "to": "historical", "to_name": "Historical",
         "text": "6 rivals inside 3 mi — is a full 12.6k anchor realistic here?", "cites": ["comp.count_3mi"]},
        {"mid": "b2", "type": "QUESTION", "type_color": "#3b82f6", "round": 1,
         "sender": "local_market", "sender_name": "Local-Market", "sender_emoji": "\U0001F3D9️", "sender_color": "#16a34a",
         "to": "competition", "to_name": "Competition",
         "text": "How many of those six are true express vs self-serve bays?", "cites": ["comp.types"]},
        {"mid": "b3", "type": "REVISE", "type_color": "#f59e0b", "round": 1,
         "sender": "historical", "sender_name": "Historical", "sender_emoji": "\U0001F4CA", "sender_color": "#2563eb",
         "to": None, "to_name": None,
         "text": "Fair — trimming the anchor to ~11.8k for local saturation.", "cites": ["hist.cluster_wash", "comp.count_3mi"]},
        {"mid": "b4", "type": "REQUEST", "type_color": "#8b5cf6", "round": 1,
         "sender": "finance", "sender_name": "Finance", "sender_emoji": "\U0001F4B0", "sender_color": "#ea580c",
         "to": "capacity", "to_name": "Capacity",
         "text": "Re-price CAPEX at 120 ft so I can lock the payback month.", "cites": ["cap.tunnel_ft"]},
        {"mid": "c1", "type": "PUBLISH", "type_color": "#64748b", "round": 2,
         "sender": "capacity", "sender_name": "Capacity", "sender_emoji": "\U0001F3D7️", "sender_color": "#9333ea",
         "to": None, "to_name": None,
         "text": "120 ft build pencils to ~$1.5M CAPEX from the enriched proforma fit.", "cites": ["cap.capex"]},
        {"mid": "c2", "type": "REVISE", "type_color": "#f59e0b", "round": 2,
         "sender": "finance", "sender_name": "Finance", "sender_emoji": "\U0001F4B0", "sender_color": "#ea580c",
         "to": None, "to_name": None,
         "text": "With 11.8k washes + $1.5M CAPEX, 5-yr net ~$3.1M, breakeven month 31.", "cites": ["fin.net_5yr", "fin.breakeven"]},
        {"mid": "c3", "type": "ENDORSE", "type_color": "#22c55e", "round": 2,
         "sender": "local_market", "sender_name": "Local-Market", "sender_emoji": "\U0001F3D9️", "sender_color": "#16a34a",
         "to": "finance", "to_name": "Finance",
         "text": "55% membership share sustains that revenue — I back the number.", "cites": ["lm.membership_share"]},
        {"mid": "v1", "type": "VOTE", "type_color": "#0f172a", "round": 3,
         "sender": "historical", "sender_name": "Historical", "sender_emoji": "\U0001F4CA", "sender_color": "#2563eb",
         "to": None, "to_name": None, "text": "Build — the trimmed anchor still clears the bar.", "cites": ["hist.cluster_wash"]},
        {"mid": "v2", "type": "VOTE", "type_color": "#0f172a", "round": 3,
         "sender": "competition", "sender_name": "Competition", "sender_emoji": "\U0001F6F0️", "sender_color": "#dc2626",
         "to": None, "to_name": None, "text": "Conditional — fine if the anchor stays trimmed.", "cites": ["comp.count_3mi"]},
        {"mid": "v3", "type": "VOTE", "type_color": "#0f172a", "round": 3,
         "sender": "local_market", "sender_name": "Local-Market", "sender_emoji": "\U0001F3D9️", "sender_color": "#16a34a",
         "to": None, "to_name": None, "text": "Build.", "cites": ["lm.membership_share"]},
        {"mid": "v4", "type": "VOTE", "type_color": "#0f172a", "round": 3,
         "sender": "capacity", "sender_name": "Capacity", "sender_emoji": "\U0001F3D7️", "sender_color": "#9333ea",
         "to": None, "to_name": None, "text": "Build — capacity is not the constraint.", "cites": ["cap.tunnel_ft"]},
        {"mid": "v5", "type": "VOTE", "type_color": "#0f172a", "round": 3,
         "sender": "finance", "sender_name": "Finance", "sender_emoji": "\U0001F4B0", "sender_color": "#ea580c",
         "to": None, "to_name": None, "text": "Build — payback inside 3 years.", "cites": ["fin.breakeven"]},
    ],
    "belief_history": {
        "historical":   [{"round": 0, "confidence": 0.72, "key_number": 12588, "lean": "Build"},
                         {"round": 1, "confidence": 0.70, "key_number": 11800, "lean": "Build"},
                         {"round": 3, "confidence": 0.71, "key_number": 11800, "lean": "Build"}],
        "competition":  [{"round": 0, "confidence": 0.55, "key_number": 6, "lean": "Conditional"},
                         {"round": 1, "confidence": 0.60, "key_number": 6, "lean": "Conditional"},
                         {"round": 3, "confidence": 0.58, "key_number": 6, "lean": "Conditional"}],
        "local_market": [{"round": 0, "confidence": 0.60, "key_number": 0.55, "lean": "Build"},
                         {"round": 2, "confidence": 0.64, "key_number": 0.55, "lean": "Build"}],
        "capacity":     [{"round": 0, "confidence": 0.68, "key_number": 120, "lean": "Build"},
                         {"round": 2, "confidence": 0.71, "key_number": 120, "lean": "Build"}],
        "finance":      [{"round": 0, "confidence": 0.50, "key_number": None, "lean": None},
                         {"round": 2, "confidence": 0.74, "key_number": 3100000, "lean": "Build"},
                         {"round": 3, "confidence": 0.76, "key_number": 3100000, "lean": "Build"}],
    },
}


if __name__ == "__main__":
    print(build_chamber_html(DEMO_DATA))
