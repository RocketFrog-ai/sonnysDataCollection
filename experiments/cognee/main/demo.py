"""
Cognee demo — turn the `context/` folder into a queryable knowledge graph.

A 30-minute live walkthrough, in seven acts:

  1  INGEST     raw files in. No schema, no chunking config, no pipeline code.
  2  COGNIFY    an LLM reads them and emits entities + typed relationships -> a graph.
  3  INSPECT    what actually got built: node/edge counts, sample triplets.
  4  SEARCH     the same question through 5 retrieval modes, side by side.
  5  MULTI-HOP  a question plain vector RAG structurally cannot answer.
  6  VISUALIZE  the graph as an interactive HTML page.
  7  RECEIPTS   measured latency + token cost. The honest downside.

Run:
    ./run.sh                 # full demo
    ./run.sh --skip-cognify  # re-query an existing graph (fast, for re-runs)
    ./run.sh --act 4         # jump to one act

Everything is printed with timings so the audience sees the real cost of each step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # experiments/cognee
CONTEXT_DIR = ROOT / "context"
REPO_ROOT = ROOT.parent.parent          # sonnysDataCollection
OUT_DIR = ROOT / "out"
DATASET = "sonnys_cdh"

# ----------------------------------------------------------------------------
# Provider config. MUST be set before `import cognee`.
# ----------------------------------------------------------------------------


def _load_repo_env() -> None:
    """Pull AZURE_/GEMINI_ keys out of the repo-root .env without a dependency."""
    envf = REPO_ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def configure_provider() -> str:
    """
    Pick an LLM backend and wire cognee to it.

    LLM: Azure `gpt-4o` (the repo's normal backend, same creds
    `app/pnl_analysis/insights/` uses) when reachable, else Gemini. Azure sits
    behind a private endpoint and returns 403 "Public access is disabled" when
    you are off the corporate network, so we probe before committing to it.
    Force one with COGNEE_BACKEND=azure|gemini.

    EMBEDDINGS: always local (fastembed / ONNX, no network). Two reasons, both
    discovered the hard way —
      * our Azure resource has a gpt-4o deployment but NO embedding deployment
        (`404 DeploymentNotFound` for text-embedding-3-*), and
      * Gemini's free-tier embedding quota 429s within seconds of cognify's
        fan-out.
    Local embedding also makes the demo reproducible offline, which removes a
    whole class of live-demo risk.
    """
    _load_repo_env()
    choice = os.environ.get("COGNEE_BACKEND", "").lower()

    os.environ.update(
        EMBEDDING_PROVIDER="fastembed",
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        EMBEDDING_DIMENSIONS="384",
        EMBEDDING_MAX_TOKENS="512",
    )

    if choice != "gemini" and _azure_reachable():
        dep = os.environ["AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"]
        os.environ.update(
            LLM_PROVIDER="custom",
            LLM_MODEL=f"azure/{dep}",
            LLM_ENDPOINT=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            LLM_API_KEY=os.environ["AZURE_OPENAI_API_KEY"],
            LLM_API_VERSION=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        llm = f"azure/{dep}"
    else:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            sys.exit("No reachable LLM backend: Azure is private-endpoint only "
                     "(are you on the network?) and GEMINI_API_KEY is unset.")
        os.environ.update(
            LLM_PROVIDER="gemini",
            LLM_MODEL="gemini/gemini-2.5-flash",
            LLM_API_KEY=key,
        )
        llm = "gemini-2.5-flash"

    # Gemini free tier and Azure PTU both throttle; cognify fans out hard.
    os.environ.setdefault("LLM_RATE_LIMIT_ENABLED", "true")
    os.environ.setdefault("LLM_RATE_LIMIT_REQUESTS", "60")
    os.environ.setdefault("LLM_RATE_LIMIT_INTERVAL", "60")

    backend = f"{llm} (LLM) + BAAI/bge-small-en-v1.5 (local embeddings)"

    # Single-user demo: skip cognee's multi-tenant ACL layer.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    # cognee logs every pipeline step at INFO. Unreadable on a projector.
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    # cognee 1.x turns on session memory by DEFAULT. It makes every search
    # aware of the previous one, so a side-by-side comparison degenerates into
    # "It seems I already answered that." Each act must be independent.
    os.environ.setdefault("CACHING", "false")
    # Keep all state inside this experiment folder, not the user's home dir.
    os.environ.setdefault("COGNEE_SYSTEM_DIRECTORY_PATH", str(ROOT / ".cognee_system"))
    os.environ.setdefault("COGNEE_DATA_DIRECTORY_PATH", str(ROOT / ".cognee_data"))
    return backend


def _azure_reachable() -> bool:
    import json as _json
    import urllib.error
    import urllib.request

    ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    dep = os.environ.get("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "")
    ver = os.environ.get("AZURE_OPENAI_API_VERSION", "")
    if not all((ep, key, dep, ver)):
        return False
    ver = ver or "2024-02-15-preview"
    url = f"{ep}/openai/deployments/{dep}/chat/completions?api-version={ver}"
    body = _json.dumps({"messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"api-key": key, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=8)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Cost meter — wraps litellm so we can show what cognee actually spends.
# ----------------------------------------------------------------------------


class Meter:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embed_calls = 0

    def install(self) -> None:
        import litellm

        meter = self

        def wrap(fn, is_embed=False):
            async def inner(*a, **kw):
                res = await fn(*a, **kw)
                if is_embed:
                    meter.embed_calls += 1
                else:
                    meter.calls += 1
                    u = getattr(res, "usage", None)
                    if u:
                        meter.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                        meter.completion_tokens += getattr(u, "completion_tokens", 0) or 0
                return res
            return inner

        litellm.acompletion = wrap(litellm.acompletion)
        litellm.aembedding = wrap(litellm.aembedding, is_embed=True)

    def snapshot(self) -> dict:
        return dict(llm_calls=self.calls, embed_calls=self.embed_calls,
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens)

    def delta(self, before: dict) -> dict:
        now = self.snapshot()
        return {k: now[k] - before[k] for k in now}


METER = Meter()

# ----------------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------------

BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
TIMINGS: list[tuple[str, float, dict]] = []


def act(n: int, title: str, subtitle: str = "") -> None:
    print(f"\n{CYAN}{'━' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  ACT {n}  ·  {title}{RESET}")
    if subtitle:
        print(f"{DIM}  {subtitle}{RESET}")
    print(f"{CYAN}{'━' * 78}{RESET}")


def step(msg: str) -> None:
    print(f"\n{BOLD}▸ {msg}{RESET}")


def note(msg: str) -> None:
    print(f"{DIM}  {msg}{RESET}")


def wrap_text(s: str, width: int = 74, indent: str = "    ") -> str:
    import textwrap
    out = []
    for para in str(s).split("\n"):
        out.extend(textwrap.wrap(para, width, initial_indent=indent,
                                 subsequent_indent=indent) or [""])
    return "\n".join(out)


class timed:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.before = METER.snapshot()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        d = METER.delta(self.before)
        TIMINGS.append((self.label, dt, d))
        print(f"{GREEN}  ✔ {self.label}: {dt:.1f}s "
              f"· {d['llm_calls']} LLM calls · {d['embed_calls']} embed calls "
              f"· {d['prompt_tokens'] + d['completion_tokens']:,} tokens{RESET}")
        return False


# ----------------------------------------------------------------------------
# The demo questions. Chosen to make graph-vs-RAG differences visible.
# ----------------------------------------------------------------------------

# Act 4: one question, five retrieval modes.
HEADLINE_Q = "How do I correctly classify a member wash versus a retail wash?"

# Act 5: multi-hop. The answer is not in any single chunk — it requires stitching
# the join chain (§8) to the filter rules (§2) to the refund rule (§3).
MULTIHOP_QS = [
    "Trace the full join path from a completed transaction to the current list "
    "price of the customer's membership package, and state every filter that "
    "must be applied along the way.",
    "A monthly wash-count query returns numbers ~10x too high. Based on the "
    "documented rules, list the specific mistakes that could cause this.",
    "What is the relationship between washbook_billing_history and signup "
    "payments, and why does that matter for revenue reporting?",
]

# Act 5b: cross-document. Does the graph connect the SKILL.md rules to the CDH
# data-definition PDF, which was written by different people for a different purpose?
CROSS_DOC_Q = (
    "The data-definitions document lists a metric called 'car wash count by members'. "
    "Using the SQL guide, give the exact logic that implements it: which flag "
    "combination defines a member wash, which filters must be applied, and how "
    "refunds are handled."
)


# ----------------------------------------------------------------------------
# Acts
# ----------------------------------------------------------------------------


async def act1_ingest(cognee, reset: bool) -> list[Path]:
    act(1, "INGEST", "Raw files in. No schema. No chunking config. No pipeline code.")

    files = sorted(p for p in CONTEXT_DIR.iterdir()
                   if p.suffix.lower() in {".md", ".pdf", ".txt", ".csv", ".json"})
    for f in files:
        kb = f.stat().st_size / 1024
        print(f"    {f.name:<46} {kb:>8.0f} KB")
    note(f"{len(files)} files — a markdown SQL playbook and a PDF of metric definitions.")
    note("Different authors, different formats, no shared IDs. That is the point.")

    if reset:
        step("Pruning previous state (fresh graph)")
        with timed("prune"):
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

    step(f"cognee.add(...) -> dataset '{DATASET}'")
    with timed("add"):
        await cognee.add([str(f) for f in files], dataset_name=DATASET)
    note("add() stores + classifies raw bytes and picks a loader per file type")
    note("(pypdf for the PDF, text for the markdown). Nearly free — no graph yet.")
    return files


async def act2_cognify(cognee) -> None:
    act(2, "COGNIFY", "An LLM reads the documents and emits entities + typed edges.")
    note("This is the expensive step — and the whole value proposition.")
    note("Pipeline: chunk -> extract graph -> summarize -> embed -> persist.")

    step("cognee.cognify()  (this is where the minutes and tokens go)")
    with timed("cognify"):
        ticker = asyncio.create_task(_heartbeat())
        try:
            # Default chunk size swallows each document whole (2 chunks total),
            # which yields a thin graph and makes the CHUNKS-vs-GRAPH contrast
            # meaningless. Smaller chunks -> more extraction passes -> a graph
            # dense enough to actually be worth looking at.
            await cognee.cognify(datasets=[DATASET], chunk_size=512)
        finally:
            ticker.cancel()
            print()  # clear the heartbeat line

    # memify() is cognee's second-pass enrichment: it builds derived structure
    # ON TOP of the graph cognify produced. Here it embeds each triplet so
    # TRIPLET_COMPLETION can retrieve edges directly, not just entities.
    step("cognee.memify(create_triplet_embeddings)  — a second pass over the graph")
    note("Without this, SearchType.TRIPLET_COMPLETION raises NoDataError. Worth")
    note("knowing: some search modes need enrichment beyond plain cognify.")
    try:
        from cognee.memify_pipelines.create_triplet_embeddings import (
            create_triplet_embeddings)
        from cognee.modules.users.methods import get_default_user

        with timed("memify"):
            await create_triplet_embeddings(user=await get_default_user(),
                                            dataset=DATASET)
    except Exception as exc:
        print(f"{YELLOW}    ~ memify skipped: {type(exc).__name__}: {exc}{RESET}")
        note("TRIPLET_COMPLETION will report NoDataError in Act 4 — that is the")
        note("honest behaviour, not a demo bug.")


async def _heartbeat() -> None:
    """Live ticker so a 3-minute cognify doesn't look like a frozen terminal."""
    t0 = time.time()
    try:
        while True:
            s = METER.snapshot()
            print(f"\r{DIM}    …{time.time() - t0:5.0f}s  "
                  f"{s['llm_calls']:>3} LLM calls  {s['embed_calls']:>3} embeds  "
                  f"{s['prompt_tokens'] + s['completion_tokens']:>7,} tokens{RESET}",
                  end="", flush=True)
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass


async def act3_inspect(cognee) -> dict:
    act(3, "INSPECT", "What actually got built. Not a black box — read the graph.")

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()

    by_type: dict[str, int] = {}
    for _nid, attrs in nodes:
        t = (attrs or {}).get("type") or (attrs or {}).get("__type__") or "Unknown"
        by_type[str(t)] = by_type.get(str(t), 0) + 1

    step(f"Graph: {len(nodes):,} nodes · {len(edges):,} edges")
    print(f"\n    {BOLD}Node types{RESET}")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1])[:12]:
        print(f"      {t:<34} {c:>6,}")

    rel_counts: dict[str, int] = {}
    for e in edges:
        rel = e[2] if len(e) > 2 else "?"
        rel_counts[str(rel)] = rel_counts.get(str(rel), 0) + 1
    print(f"\n    {BOLD}Relationship types (top 15 of {len(rel_counts)}){RESET}")
    for r, c in sorted(rel_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"      {r:<34} {c:>6,}")

    # Sample domain triplets — show the audience it learned OUR vocabulary,
    # not a generic ontology.
    def clean(v: object, width: int = 38) -> str:
        s = " ".join(str(v).split())          # PDF text carries hard newlines
        return s[: width - 1] + "…" if len(s) > width else s

    name_of, type_of = {}, {}
    for nid, attrs in nodes:
        a = attrs or {}
        name_of[str(nid)] = clean(a.get("name") or a.get("text") or nid)
        type_of[str(nid)] = str(a.get("type") or "")

    # The interesting edges are Entity->Entity. `contains` (chunk->entity) and
    # `is_a` (entity->type) are plumbing — they'd crowd out the real content.
    plumbing = {"contains", "is_a", "is_part_of", "made_from", "originates_from"}
    picked = [
        (name_of.get(str(e[0]), ""), str(e[2]), name_of.get(str(e[1]), ""))
        for e in edges
        if len(e) >= 3
        and str(e[2]) not in plumbing
        and type_of.get(str(e[0])) == "Entity"
        and type_of.get(str(e[1])) == "Entity"
    ]

    print(f"\n    {BOLD}Sample triplets it learned from our documents{RESET}")
    if picked:
        for s, r, t in picked[:20]:
            print(f"      {s:<38} {DIM}--{r}->{RESET}  {t}")
        if len(picked) > 20:
            note(f"(+{len(picked) - 20} more entity-to-entity edges)")
    else:
        note("(no entity-to-entity edges — extraction produced only plumbing edges)")

    note("Nobody wrote an ontology. These entity + edge types were inferred.")
    return dict(nodes=len(nodes), edges=len(edges), node_types=by_type,
                rel_types=len(rel_counts))


async def act4_search_modes(cognee) -> None:
    from cognee import SearchType

    act(4, "SEARCH — five modes, one question",
        f'Q: "{HEADLINE_Q}"')

    modes = [
        (SearchType.CHUNKS, "CHUNKS",
         "Raw vector hits. No LLM. What classic RAG retrieves before it writes."),
        (SearchType.RAG_COMPLETION, "RAG_COMPLETION",
         "Classic vector RAG: top-k chunks -> LLM. The baseline to beat."),
        (SearchType.GRAPH_COMPLETION, "GRAPH_COMPLETION",
         "Retrieves a connected subgraph, not loose chunks. The headline feature."),
        (SearchType.TRIPLET_COMPLETION, "TRIPLET_COMPLETION",
         "Answers strictly from subject-predicate-object edges. Most auditable."),
        (SearchType.SUMMARIES, "SUMMARIES",
         "Hierarchical summaries built during cognify. Cheap orientation."),
    ]

    for st, name, why in modes:
        step(f"SearchType.{name}")
        note(why)
        try:
            with timed(f"search:{name}"):
                res = await cognee.search(query_text=HEADLINE_Q, query_type=st,
                                          datasets=[DATASET])
            _print_result(res)
        except Exception as exc:
            print(f"{RED}    ! {name} failed: {type(exc).__name__}: {exc}{RESET}")

    note("Same question, same corpus. Compare what each mode grounds its answer in.")


async def act5_multihop(cognee) -> None:
    from cognee import SearchType

    act(5, "MULTI-HOP", "Questions whose answer lives in no single chunk.")
    note("Vector RAG retrieves passages that look like the question. When the answer")
    note("is a CHAIN across sections, similarity search has nothing to match on.")

    for q in MULTIHOP_QS:
        step(q)
        try:
            with timed("graph_completion"):
                res = await cognee.search(query_text=q,
                                          query_type=SearchType.GRAPH_COMPLETION,
                                          datasets=[DATASET])
            _print_result(res)
        except Exception as exc:
            print(f"{RED}    ! failed: {type(exc).__name__}: {exc}{RESET}")

    step("Cross-document link (this is the real test)")
    note("The PDF and the markdown were written separately, share no IDs, and use")
    note("different words for the same things. Can the graph join them anyway?")
    print(f"    {DIM}Q: {CROSS_DOC_Q}{RESET}")
    try:
        with timed("cross_doc"):
            res = await cognee.search(query_text=CROSS_DOC_Q,
                                      query_type=SearchType.GRAPH_COMPLETION,
                                      datasets=[DATASET])
        _print_result(res)
    except Exception as exc:
        print(f"{RED}    ! failed: {type(exc).__name__}: {exc}{RESET}")


async def act6_visualize(cognee) -> Path | None:
    act(6, "VISUALIZE", "The graph as an interactive page you can hand to a stakeholder.")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "knowledge_graph.html"
    try:
        with timed("visualize"):
            await cognee.visualize_graph(str(out))
        if out.exists():
            _inline_assets(out)
            print(f"    {GREEN}{out}{RESET}  ({out.stat().st_size / 1024:.0f} KB)")
            note("Open it: drag into a browser, or `open out/knowledge_graph.html`")
            return out
    except Exception as exc:
        print(f"{RED}    ! visualize failed: {type(exc).__name__}: {exc}{RESET}")
    return None


def _inline_assets(page: Path) -> None:
    """
    cognee's visualizer loads d3 and Google Fonts from CDNs. On a locked-down
    demo network that renders a blank page — the single most embarrassing way
    for this to fail live. Inline d3 and drop the font links so the file works
    offline.
    """
    import re
    import urllib.request

    html = page.read_text()
    m = re.search(r'<script\s+src="(https?://[^"]*d3[^"]*)"\s*></script>', html)
    if m:
        d3 = _vendored_d3(m.group(1))
        if d3:
            html = html.replace(m.group(0), f"<script>{d3}</script>")
            note("d3 inlined — the page renders with no network access.")
        else:
            print(f"{YELLOW}    ~ could not inline d3; the page will need "
                  f"{m.group(1)} reachable when you open it.{RESET}")
    html = re.sub(r'<link[^>]*fonts\.(googleapis|gstatic)\.com[^>]*>', "", html)
    page.write_text(html)


def _vendored_d3(primary: str) -> str | None:
    """
    Fetch d3 once and cache it next to the demo.

    Note the User-Agent: d3js.org returns 403 to urllib's default
    `Python-urllib/3.x`, which looks exactly like a blocked network and cost us
    a debugging detour. jsdelivr and cdnjs are listed as fallbacks in case the
    presenting network blocks one of them.
    """
    import urllib.request

    cache = HERE / "_vendor_d3.v7.min.js"
    if cache.exists() and cache.stat().st_size > 100_000:
        note("d3 loaded from local cache (no network needed).")
        return cache.read_text()

    for url in (primary,
                "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
                "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                js = r.read().decode("utf-8", "replace")
            if len(js) > 100_000:
                cache.write_text(js)
                return js
        except Exception:
            continue
    return None


def act7_receipts(backend: str, stats: dict) -> None:
    act(7, "RECEIPTS", "The honest cost. Every number below was measured just now.")

    print(f"    {BOLD}{'step':<26}{'wall':>9}{'LLM':>7}{'embed':>8}{'tokens':>12}{RESET}")
    total_t = total_c = total_e = total_tok = 0.0
    for label, dt, d in TIMINGS:
        tok = d["prompt_tokens"] + d["completion_tokens"]
        print(f"    {label:<26}{dt:>8.1f}s{d['llm_calls']:>7}{d['embed_calls']:>8}{tok:>12,}")
        total_t += dt
        total_c += d["llm_calls"]
        total_e += d["embed_calls"]
        total_tok += tok
    print(f"    {'-' * 62}")
    print(f"    {BOLD}{'TOTAL':<26}{total_t:>8.1f}s{int(total_c):>7}{int(total_e):>8}{int(total_tok):>12,}{RESET}")

    cognify = next((t for lbl, t, _ in TIMINGS if lbl == "cognify"), 0.0)
    searches = [(lbl, t) for lbl, t, _ in TIMINGS if lbl.startswith(("search:", "graph_", "cross_"))]
    avg_q = sum(t for _, t in searches) / max(len(searches), 1)

    if total_e == 0:
        note("embed count is 0 because embeddings run locally (fastembed/ONNX) and never")
        note("touch litellm — they cost CPU, not tokens. That is the point of running local.")

    print(f"\n    {BOLD}What this means{RESET}")
    print(f"      backend            {backend}")
    print(f"      corpus             2 documents, ~30k characters")
    print(f"      graph built        {stats.get('nodes', 0):,} nodes / {stats.get('edges', 0):,} edges")
    print(f"      build cost         {cognify:.0f}s one-time for this corpus")
    print(f"      query cost         {avg_q:.1f}s average, ~1-3 LLM calls each")
    if cognify and total_tok:
        print(f"      scaling            ingest cost is ~linear in corpus size;")
        print(f"                         query cost is flat regardless of corpus size")

    json_out = OUT_DIR / "run_metrics.json"
    OUT_DIR.mkdir(exist_ok=True)
    json_out.write_text(json.dumps(
        dict(backend=backend, graph=stats,
             steps=[dict(step=l, seconds=round(t, 2), **d) for l, t, d in TIMINGS]),
        indent=2))
    note(f"Machine-readable copy: {json_out}")


def _print_result(res) -> None:
    if res is None:
        print("    (no result)")
        return
    items = res if isinstance(res, list) else [res]
    if not items:
        print("    (empty)")
        return
    for item in items[:3]:
        if isinstance(item, dict):
            txt = item.get("text") or item.get("content") or json.dumps(item)[:600]
        else:
            txt = str(item)
        txt = txt.strip()
        if len(txt) > 1600:
            txt = txt[:1600] + " …"
        print(wrap_text(txt))
        if len(items) > 1:
            print()
    if len(items) > 3:
        note(f"(+{len(items) - 3} more results)")


# ----------------------------------------------------------------------------


async def main() -> None:
    ap = argparse.ArgumentParser(description="Cognee live demo")
    ap.add_argument("--skip-cognify", action="store_true",
                    help="reuse the existing graph; skip ingest+build (fast re-runs)")
    ap.add_argument("--act", type=int, default=None, help="run a single act (1-7)")
    ap.add_argument("--keep", action="store_true",
                    help="do not prune; add to the existing graph")
    args = ap.parse_args()

    backend = configure_provider()
    METER.install()

    import cognee

    print(f"\n{BOLD}COGNEE DEMO{RESET} · {DIM}car-wash POS knowledge graph{RESET}")
    print(f"{DIM}  cognee {cognee.get_cognee_version()}  ·  backend: {backend}{RESET}")
    print(f"{DIM}  corpus: {CONTEXT_DIR}{RESET}")

    only = args.act
    stats: dict = {}

    if not args.skip_cognify and only in (None, 1, 2):
        if only in (None, 1):
            await act1_ingest(cognee, reset=not args.keep)
        if only in (None, 2):
            await act2_cognify(cognee)

    if only in (None, 3):
        stats = await act3_inspect(cognee)
    if only in (None, 4):
        await act4_search_modes(cognee)
    if only in (None, 5):
        await act5_multihop(cognee)
    if only in (None, 6):
        await act6_visualize(cognee)
    if only in (None, 7) or only is None:
        act7_receipts(backend, stats)

    print(f"\n{BOLD}{GREEN}Done.{RESET} {DIM}Read README.md for the "
          f"benefits / limits / where-this-fits writeup.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
