# Cognee — running demo

A working demo of [cognee](https://github.com/topoteretes/cognee) built on **our own documents**:
the car-wash POS SQL playbook (`SKILL.md`) and the CDH data-definitions PDF. It ingests them,
builds a knowledge graph, and answers questions that plain vector RAG cannot.

```bash
cd experiments/cognee
./main/run.sh                 # full demo, seven acts, ~90 seconds end to end
./main/run.sh --skip-cognify  # re-query an existing graph (seconds) — rehearse with this
./main/run.sh --act 4         # jump to one act
```

Isolated venv at `.venv-cognee` (auto-created on first run) — it does **not** touch conda
`sonnys`, so the pinned scipy/numpy and `scripts/smoke.sh` are unaffected.

**Verified working end to end** on 2026-08-04: 7/7 acts, exit 0, 87s wall, 68 LLM calls,
126k tokens, on `azure/gpt-4o` + local embeddings.

---

## What cognee actually is

One sentence: **cognee replaces your RAG pipeline with a memory layer that stores a knowledge
graph and a vector index over the same content, and keeps them in sync.**

```
                    ┌──────────── cognee.add() ────────────┐
  PDFs, markdown,   │ loader per filetype, dedup, store    │
  CSV, code, chat ─▶│ (no LLM, cheap)                      │
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌──────────── cognee.cognify() ────────┐
                    │ chunk → LLM extracts entities and    │
                    │ typed relationships → summarize →    │
                    │ embed → write to BOTH stores         │  ← the expensive step
                    └──────────────────┬───────────────────┘
                                       ▼
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
        graph store (Kuzu default)                  vector store (LanceDB default)
        entities + edges, traversable               chunks + embeddings, similarity
                 └─────────────────────┬─────────────────────┘
                                       ▼
                            cognee.search(query, SearchType.X)
```

The core claim: for questions whose answer is a **chain of facts spread across documents**,
retrieving a connected subgraph beats retrieving the top-k most similar paragraphs.

### The pieces you get for free

| | |
|---|---|
| **Loaders** | pdf (pypdf + docling), text, csv, image, audio, video — auto-selected |
| **Graph store** | Kuzu embedded by default; Neo4j / FalkorDB / Memgraph swappable |
| **Vector store** | LanceDB embedded by default; pgvector / Qdrant / Weaviate / Milvus swappable |
| **Relational** | SQLite by default; Postgres swappable |
| **LLM** | anything litellm speaks — OpenAI, Azure, Gemini, Anthropic, Ollama |
| **18 search modes** | see below |
| **Visualizer** | `visualize_graph()` → self-contained interactive HTML |
| **Ontologies** | optionally constrain extraction to a schema you supply |
| **`memify()`** | a second pass that builds higher-order structure over the graph |

### The search modes that matter

| SearchType | What it does | Use it when |
|---|---|---|
| `CHUNKS` | raw vector hits, no LLM | debugging retrieval; showing the RAG baseline |
| `RAG_COMPLETION` | classic top-k → LLM | the baseline you are trying to beat |
| `GRAPH_COMPLETION` | retrieves a connected subgraph → LLM | **the default. Multi-hop questions.** |
| `TRIPLET_COMPLETION` | answers only from subject-predicate-object edges | you need auditability |
| `GRAPH_COMPLETION_COT` | iterative reasoning over the graph | hard questions, higher cost |
| `SUMMARIES` | hierarchical summaries built at cognify time | orientation, "what is in here" |
| `CYPHER` / `NATURAL_LANGUAGE` | direct graph query | you know the graph shape |
| `TEMPORAL` | time-aware retrieval | event sequences |
| `CODE` / `CODING_RULES` | code-graph specific | repo Q&A |

The demo runs five of these on the **same question** so the difference is visible, not asserted.

---

## Benefits

**1. Multi-hop retrieval genuinely works.** Ask "trace the join path from a completed
transaction to the current list price of a customer's membership package, and every filter
along the way" — that answer lives in §8 (join snippets), §2 (magic numbers), and §3
(refunds) of `SKILL.md`. No single chunk contains it. Vector similarity has nothing to
match against, because the question does not *look like* any one passage. Graph traversal
walks the chain, and in the measured run it returned the correct 9-step join path with the
right filter at each step.

> **Say this honestly in the room.** On the *simple* lookup question ("member vs retail
> wash?") `RAG_COMPLETION` and `GRAPH_COMPLETION` both got it right — the rule sits in one
> paragraph, so similarity search finds it. Graph used **2,735 tokens vs RAG's 5,722**, but
> the answers were equivalent. The separation appears on the multi-hop questions in Act 5.
> If you claim graph beats RAG on everything, someone will test it on an easy question and
> you will lose the room.

**2. Cross-document joins with no shared keys.** `SKILL.md` and the CDH PDF were written by
different people, in different formats, using different words for the same things. Cognee
links them at the *entity* level — both mention member washes, `washbook_*`, transaction
filters — so a question can span both without anyone writing a mapping.

**3. Ingestion is genuinely two lines.** `add()` then `cognify()`. No chunking strategy, no
embedding model choice, no vector-store schema, no retrieval tuning. Compare to the RAG
plumbing we would otherwise hand-write and maintain.

**4. Query cost is flat as the corpus grows.** Building the graph is roughly linear in corpus
size and paid once. Querying stays ~1–3 LLM calls regardless of whether the graph holds
2 documents or 2,000.

**5. It is inspectable, not a black box.** You can dump every node and edge, print triplets,
and render an interactive HTML graph. When an answer is wrong you can see *which* extracted
edge was wrong — much better than "the retriever returned bad chunks, somehow."

**6. Swappable everything.** Embedded defaults (Kuzu + LanceDB + SQLite) mean zero infra to
demo; the same code points at Neo4j + pgvector in production by changing env vars.

**7. It fits our existing stack.** Runs on our Azure `gpt-4o` deployment via litellm — same
creds `app/pnl_analysis/insights/` already uses. Nothing new to procure. Embeddings can run
locally on CPU, so the only per-query cost is the LLM call we already pay for elsewhere.

## Disadvantages — the parts to say out loud

**1. Cognify is an LLM pass over every chunk, and the cost is real.** Measured on this
corpus: 2 documents, ~30k characters, `chunk_size=512` → **~60 LLM calls / ~92k tokens in
~16 seconds**. That is cheap here only because the corpus is tiny. It scales roughly
linearly, you re-pay it on every re-ingest, and chunk size is a direct cost multiplier — at
cognee's *default* chunk size the same corpus took **5 calls** and produced a much thinner
49-node graph instead of ~150–200. Quality and cost are the same dial, and the cheap setting
still looks like it worked.

**1b. `memify` is slower than `cognify` here** (33s vs 16s) even though it makes zero LLM
calls — it embeds every triplet locally on CPU. Budget for it if you want
`TRIPLET_COMPLETION`.

**2. Extraction quality is the ceiling.** The graph is only as good as what the LLM pulled
out. Table-heavy PDFs extract messily — the CDH PDF's tables come out as text soup, and the
entities inferred from them are noisier than the ones from clean markdown. Garbage in,
confidently-structured garbage out.

**3. No schema unless you impose one, and it is not deterministic.** Without an ontology,
entity and relation types are whatever the LLM felt like emitting that run. Three runs over
the *identical* corpus produced **154, 202, and 207 nodes** with 69–95 distinct relationship
types — `joined_with`, `join_with`, `linked_to`, `linked_with`, `linked_by`, `associated_with`
all coexist as separate edge types meaning roughly the same thing. Roughly 2/3 of all edges
are plumbing (`contains`, `is_a`). Supplying an ontology fixes the sprawl — but then you are
back to writing schemas, which was the thing it was supposed to save you.

*Do not promise a stakeholder a specific node count.* Rerun the demo and it changes.

**4. It cannot do arithmetic on our panel.** This is the important one for us. Cognee is a
retrieval-and-reasoning layer over *text*. It does not compute wash counts, fit models, or
run our P&L math. It is not a replacement for `proforma/`, and pointing it at the panel CSV
would produce a graph of *strings that look like numbers*, not a queryable metric store.

**5. Young, fast-moving project, with sharp edges you will hit.** 1.x, with breaking API
changes between minors (1.0 renamed the primary API to `remember`/`recall`/`forget` while
keeping `add`/`cognify`/`search` working). Three defaults bit us building this demo:

- **Session memory is ON by default.** Consecutive searches see each other, so the second
  question gets answered with *"It seems I already provided that in the previous response."*
  For a side-by-side comparison this is fatal. `CACHING=false` fixes it.
- **`TRIPLET_COMPLETION` fails out of the box** with `NoDataError` — it needs a separate
  `memify(create_triplet_embeddings)` pass that `cognify()` does not run. Not all 18 search
  modes work after ingestion alone.
- **Default chunk size swallows a whole document per chunk**, silently producing a thin
  graph that still *looks* like it worked.
- **`visualize_graph()` emits a page that loads d3 from a CDN.** On a locked-down demo
  network that is a blank screen in front of your audience. `demo.py` inlines d3 and caches
  it locally so the page is fully self-contained.

None are hard to fix; all are invisible until you look closely. Pin the version.

**6b. Debugging note worth stealing:** `d3js.org` returns **403 to urllib's default
`Python-urllib/3.x` User-Agent** while `curl` succeeds. That failure is indistinguishable
from a blocked corporate network, and it cost a detour. Send a browser User-Agent.

**6. Heavy dependency footprint.** Pulls litellm, lancedb, kuzu, docling, pypdf, sqlalchemy,
and more. This is exactly why the demo lives in its own venv — dropping it into conda
`sonnys` would risk the pinned scipy/numpy that `scripts/smoke.sh` guards.

**7. Non-deterministic answers.** Same question, two runs, different wording — and sometimes
different substance. Anything user-facing needs the same guardrails our `insights/` layer
already has: it annotates, it never alters a modelled number.

---

## Where this could actually earn its keep at Sonny's

Ranked by (value ÷ effort), honestly:

**1. SQL-tribal-knowledge assistant — the strongest fit.** `SKILL.md` exists precisely because
the POS schema is a minefield: composite keys, magic numbers, the OR-vs-AND member-wash trap
that causes 10x discrepancies. Ingest `SKILL.md` + the CDH definitions + our query library +
the `docs/` folder, and an analyst can ask "why is my member-wash count 10x off" and get a
grounded answer with the rule it violated. This is the demo, and it is already useful.

**2. Institutional memory over `docs/` + `DIVERGENCES.md` + model cards.** We have a lot of
"this looks broken but is knowingly broken" knowledge scattered across markdown. That is
exactly the shape of thing graph retrieval is good at — the answer to "why does
`expense_plan` move when scipy changes" spans `CLAUDE.md`, `docs/ENVIRONMENTS.md`, and
`docs/DIVERGENCES.md`.

**3. Proforma document mining.** `proforma_db/` compiled 179 proforma files (xlsx/xls/pdf)
into SQLite for the *numbers*. The prose around those numbers — assumptions, market
narratives, operator commentary — is currently unmined. A graph over that could answer
"which proformas justified their traffic assumption the same way, and how did those sites
actually perform?" Pair the graph's retrieval with our existing backtest numbers.

**4. Site-selection narrative context.** Feed competitor writeups, market reports, and
council-of-agents transcripts in, and use `GRAPH_COMPLETION` to assemble the qualitative
half of a site memo. The quantitative half stays in `coldstart.py`, where it belongs.

**Where it does not fit:** anything that must produce a number. The forecast, the P&L, the
cannibalization math — those stay deterministic. Cognee's role is the same as `insights/`:
**it annotates; it must never alter a modelled number.**

---

## Suggested 30-minute run of show

| min | act | what you say |
|---|---|---|
| 0–3 | 1 | Two files. Different authors, formats, no shared IDs. No schema written. |
| 3–8 | 2 | `cognify()` — the live ticker shows LLM calls climbing to ~60. **This is the cost. Name it out loud before anyone asks.** Then `memify()` — a second pass that enriches the graph. |
| 8–12 | 3 | Here is the graph: 202 nodes, 533 edges, 72 relationship types. **Read the triplets aloud** — `transactions --joined_with-> sales device`, `item --represents-> membership package`. Nobody wrote that ontology. |
| 12–18 | 4 | One question, five modes. Show `CHUNKS` (raw, 0 LLM calls) → `RAG_COMPLETION` → `GRAPH_COMPLETION`. Concede up front that RAG also gets this one right, at 2× the tokens. |
| 18–24 | 5 | **The money slide.** The 9-step join-path question. No single chunk contains that answer. Then the cross-document question. |
| 24–27 | 6 | Open the HTML graph. Let them drag nodes around — this is what makes it feel real. |
| 27–30 | 7 | The receipts table, then the "where it fits / where it doesn't" list. **End on "it annotates; it never alters a modelled number."** |

**Rehearse with `--skip-cognify` first** — it reuses the built graph and runs in seconds, so
you can practise the narration without re-paying for cognify.

---

## Configuration

**LLM** — auto-detected, no flags needed:

- **Azure `gpt-4o`** (`AZURE_OPENAI_*` from repo-root `.env`) — our normal backend, same
  creds `app/pnl_analysis/insights/` already uses. It sits behind a **private endpoint**:
  off the corporate network it returns `403 Public access is disabled`. The demo probes it
  first and silently falls back.
- **Gemini `gemini-2.5-flash`** (`GEMINI_API_KEY`) — the fallback, works over the public
  internet.

Force one with `COGNEE_BACKEND=azure` or `COGNEE_BACKEND=gemini`.

**Embeddings — always local** (`fastembed` / `BAAI/bge-small-en-v1.5`, 384-dim, ONNX, no
network). Two findings drove this, both worth repeating if anyone asks:

1. Our Azure resource has a `gpt-4o` deployment but **no embedding deployment** —
   `text-embedding-3-large/-small/ada-002` all return `404 DeploymentNotFound`. An LLM
   deployment existing does not imply the embedding one does. If you want Azure embeddings
   in production, that deployment has to be created first.
2. Gemini's free-tier embedding quota **429s within seconds** of cognify's fan-out.

Local embedding is also the right call for a live demo: no quota, no network, reproducible.
The tradeoff is honest — bge-small (384-dim) retrieves less precisely than
text-embedding-3-large (3072-dim), so production would want the Azure deployment.

## Files

```
experiments/cognee/
  context/        the corpus — SKILL.md (POS SQL playbook) + CDH data-definitions PDF
  main/demo.py    the demo, seven acts, instrumented with timing + token counts
  main/run.sh     launcher; creates .venv-cognee on first run
  out/            knowledge_graph.html + run_metrics.json (generated)
  .venv-cognee/   isolated env (gitignored) — does not touch conda `sonnys`
```
