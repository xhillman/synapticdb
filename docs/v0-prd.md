# SynapticDB v0 — Product Requirements Document

**Version:** 0.1.0 (rebuild)
**Predecessor:** `synaptic-core` v0.4.0 (kept as read-only reference at `../synaptic-core`)

---

## 1. Identity

> **SynapticDB is a single-file memory store for AI agents where recall gets
> smarter with use** — hybrid keyword + semantic search, plus spreading
> activation over a graph of associations that the library learns passively
> from how memories are stored and retrieved.

One sentence, one value proposition. SynapticDB v0 is **not** a KV store, a
session store, a framework, or a telemetry platform.

### Why a rebuild

The predecessor validated the core hypothesis — spreading activation over a
learned graph surfaces associatively-related memories that a competent
BM25 + vector + cross-encoder pipeline cannot reach (holdout run
`holdout-20260220T232040Z`: positive associative-hit delta in 12/12 runs,
mean 14.5 unique wins per 25 associative queries against a pass gate
of 10). It then buried that validated core under ~14,000 lines of
speculative machinery: a 13-table telemetry subsystem with no consumer, an
auto-optimizer for ~25 tunable parameters, a scope algebra threaded through
every method, and three overlapping storage subsystems. The winning
benchmark configuration used **none** of that — it was fusion + activation
rerank. v0 rebuilds exactly the part that won.

### Design principles (binding)

1. **Usage, not inference.** Passive learning signals record what happened
   (stored together, retrieved together). Signals that interpret meaning
   (dwell time, sentiment, response parsing) are excluded — they were the
   root cause of the predecessor's defensive complexity.
2. **Store what a public API method reads.** Every table and column must be
   read back by `recall`, `feedback`, or `stats`. No "collect now, use
   later."
3. **The benchmark is the merge gate.** A learning or ranking feature ships
   only if it improves the associative-recall benchmark (§10). Defaults
   change only with benchmark evidence.
4. **Parameter budget.** v0 has exactly the knobs in §9 (17 total, fixed
   defaults, no auto-tuning). Any feature that needs a new knob pays for it
   in review.
5. **Line budget.** Core package ≤ 2,000 lines excluding tests and bench.
   This is a design constraint, not an aspiration.

---

## 2. What ships / what does not

| Ships in v0 | |
|---|---|
| SQLite single-file backend (FTS5 + embedded vectors, numpy brute-force ANN) | ✅ |
| Hybrid retrieval: BM25 + cosine + Reciprocal Rank Fusion | ✅ |
| Spreading activation (single tier, flat parameters) | ✅ |
| Graph confidence + graceful degradation | ✅ |
| Passive learning: semantic seeding, temporal proximity, co-retrieval reinforcement | ✅ |
| Explicit feedback (positive/negative per query) | ✅ |
| Edge decay + opportunistic pruning | ✅ |
| Per-recall attribution (`via`: search / association / both) | ✅ |
| Content-hash dedupe on `store()` | ✅ |
| Optional embeddings extra (`synapticdb[embeddings]`) with MiniLM default | ✅ |
| Benchmark harness in-repo (`bench/`) | ✅ |

| Deferred | Target |
|---|---|
| Edge lifecycle beyond simple decay; recency-weighted ranking | v0.2 |
| Implicit signal `memory_in_response` (embedding-based, opt-in) | v0.2 |
| MCP server, CLI, async API variants | v0.3 |
| Memory tiers + graduation, type classification, consolidation | v0.4 (benchmark-gated) |
| Namespaces (single string column), Postgres, opt-in telemetry export | v0.5 (demand-driven) |
| KV store, session persistence, scope system, instance optimizer | Not planned |

---

## 3. Public API

Synchronous. One class, one exception hierarchy. Python ≥ 3.10.

```python
from synapticdb import SynapticDB

mem = SynapticDB(
    db_path="agent.db",          # ":memory:" supported
    embedding_fn=None,           # str -> Sequence[float]; None = built-in default (§8)
)

# --- core loop -----------------------------------------------------------
m = mem.store(
    "Client X requires SOC2 for all vendor deployments",
    metadata={"source": "call-notes"},        # optional, JSON-serializable
)                                             # -> Memory (existing one if duplicate content)

result = mem.recall(
    "deployment requirements for client X",
    top_k=10,                                 # default 10
    where=None,                               # optional {key: value} equality filter on metadata
    min_confidence=0.0,                       # drop results below this; may return fewer, or none
)                                             # -> RecallResult

result.memories                               # list[RecalledMemory], ranked
result.association_results                    # memories retrieved only through association
result.query_id                               # UUID, feeds feedback()

mem.feedback(result.query_id, positive=True)  # -> None; second call for same id raises

# --- graph ---------------------------------------------------------------
mem.connect(a_id, b_id)                       # explicit edge, weight 0.5
mem.forget(memory_id)                         # delete memory + its edges
mem.stats()                                   # -> Stats (see §3.2)

mem.close()                                   # also a context manager
```

### 3.1 Errors

```python
class SynapticError(Exception): ...           # base
class NotFoundError(SynapticError): ...       # unknown memory_id / query_id
class InvalidArgumentError(SynapticError): ...# bad input, repeat feedback
class EmbeddingError(SynapticError): ...      # embedding_fn missing/failed/wrong dim
```

No error-code/hint envelope. Exception type + message is the contract.

### 3.2 Models

```python
class Memory(BaseModel):
    id: UUID
    content: str
    metadata: dict[str, Any] = {}
    created_at: datetime           # UTC
    last_accessed_at: datetime     # UTC
    access_count: int = 0

class RecalledMemory(Memory):
    score: float                   # blended ranking strength within this query, [0, 1]
    confidence: float              # cosine evidence this memory addresses the query, [0, 1]
    via: Literal["search", "association", "both"]

**Amended 2026-07-25 — `confidence` added; threshold on it, not on `score`.**
The two answer different questions. `score` is min-max normalized within one
query, so it is comparable between that query's results and meaningless across
queries: measured, the top result's score reduced to `1 - 0.45 * maturity`,
a function of graph state rather than of the question. `confidence` is absolute
cosine similarity, so a fixed threshold works across queries — at 0.6 it kept
41/42 correct answers and rejected 12/12 questions the corpus could not answer.

An association scores low on `confidence` by construction, because weak textual
similarity is why search missed it. That is the intended dial, not a defect:
raise the threshold for direct matches only, lower it to admit associations.

class RecallResult(BaseModel):
    query_id: UUID
    memories: list[RecalledMemory]
    maturity: float                # graph confidence at query time, [0, 1]
    latency_ms: float

    @property
    def association_results(self) -> list[RecalledMemory]: ...

class Stats(BaseModel):
    memories: int
    edges: int
    edges_by_origin: dict[str, int]    # semantic | temporal | co_retrieval | explicit
    maturity: float                    # [0, 1]; expose as percentage in UIs
    db_path: str
```

Concurrency contract: a `SynapticDB` instance is **not** thread-safe; one
instance per process/thread, single writer per DB file. WAL mode is enabled
so concurrent readers of the file are safe.

---

## 4. Storage schema

Three tables + FTS5. Embeddings live on the memory row; the vector index is
an in-process numpy matrix rebuilt lazily (§7.2). Every column is read by a
public method — audit this at review time (principle 2).

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE memories (
    id                TEXT PRIMARY KEY,          -- UUID4
    content           TEXT NOT NULL,
    content_hash      TEXT NOT NULL UNIQUE,      -- sha256 of normalized content (dedupe)
    metadata          TEXT NOT NULL DEFAULT '{}',-- JSON object
    embedding         BLOB NOT NULL,             -- float32 little-endian
    created_at        TEXT NOT NULL,             -- ISO-8601 UTC
    last_accessed_at  TEXT NOT NULL,
    access_count      INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);
-- plus the standard AFTER INSERT / AFTER DELETE / AFTER UPDATE sync triggers

CREATE TABLE edges (
    id                  TEXT PRIMARY KEY,        -- deterministic: sha256(a || b)
    a                   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    b                   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight              REAL NOT NULL,           -- stored weight at last write, [0, 1]
    origin              TEXT NOT NULL,           -- 'semantic'|'temporal'|'co_retrieval'|'explicit'
    created_at          TEXT NOT NULL,
    last_reinforced_at  TEXT NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (a, b)                                -- canonical order: a < b lexicographically
);
CREATE INDEX idx_edges_a ON edges(a);
CREATE INDEX idx_edges_b ON edges(b);

CREATE TABLE queries (
    id             TEXT PRIMARY KEY,             -- UUID4
    text           TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    result_ids     TEXT NOT NULL,                -- JSON array, final ranked memory ids
    energies       TEXT NOT NULL,                -- JSON object {memory_id: final_energy}
    path_edge_ids  TEXT NOT NULL,                -- JSON array, edges traversed by activation
    feedback       INTEGER                       -- NULL | 1 | -1
);
```

`queries` is not telemetry: `feedback()` reads `result_ids`, `energies`,
and `path_edge_ids` to apply Hebbian updates; `stats()` and the benchmark
read nothing else from it. Rows older than 30 days with non-NULL feedback
(or older than 7 days with NULL) are deleted during maintenance (§6.5).

Undirected edges: `(a, b)` stored once with `a < b`; all lookups normalize
the pair.

---

## 5. Retrieval pipeline (`recall`)

```
query text
  ├── embed(query)
  ├── keyword_search: FTS5 bm25(), top 40
  ├── semantic_search: cosine vs numpy matrix, top 40
  ├── RRF fusion (k=60) -> fused ranking
  ├── graph_confidence c = confidence(graph)          (§7.1, cached)
  ├── activation: spread from top-5 fused seeds        (§5.2)
  ├── blend:  score = (1 - α)·fusion̂ + α·activation̂,  α = 0.45 · c
  ├── where-filter (metadata equality), truncate to top_k
  ├── persist query row; bump access stats on returned memories
  └── passive co-retrieval learning                    (§6.3)
```

### 5.1 Fusion

Standard RRF: `score(m) = Σ_sources 1 / (60 + rank)`. Nodes in both source
lists naturally score higher; no overlap bonus (predecessor's rule, keep it).

### 5.2 Spreading activation

Single tier. Defaults are the **locked winning config** from the
predecessor's holdout validation — they are evidence, not guesses:

```
seeds        : top 5 fusion results, initial energy = min-max-normalized fusion score
propagation  : for step in 1..5:
                 for each frontier node with energy e ≥ 0.05:
                   for each neighbor via effective edge weight w (§6.4):
                     energy[nbr] = max(energy[nbr], e · w · (1 - 0.2))
                 frontier = newly reached or increased nodes; visited nodes
                 do not re-propagate within the same recall (loop guard)
scoring      : activation_score(n) = energy[n] · (1 + 0.15 · hops(n))   for non-seeds
               activation_score(seed) = energy[seed] · (1 - 0.2)         (seed penalty)
```

The hop bonus and seed penalty are what let activation *promote
discoveries* rather than echo the fusion ranking — they were part of the
validated config and are retained deliberately.

### 5.3 Confidence-weighted blend

Replaces the predecessor's three-band merge with one formula (benchmark
must confirm parity):

```
fusion̂, activation̂ : min-max normalized to [0, 1] within this query
α = activation_blend_weight (0.45) × graph_confidence
score(m) = (1 - α) · fusion̂(m) + α · activation̂(m)     (missing source ⇒ 0 term)
via(m)   = "both" | "search" | "association"  by source membership
```

At confidence 0 this degrades exactly to hybrid search. Activation always
runs (it feeds learning and attribution) even when α is tiny.

---

## 6. Learning

Four mechanisms, one weight scale [0, 1], one decay law. All edge weights
below are *initial* weights; reinforcement moves them.

### 6.1 Semantic seeding (at `store()`)

Link the new memory to the top **3** existing memories with cosine
similarity ≥ **0.6**, initial weight **0.25**, origin `semantic`. If the
edge exists, reinforce (§6.6) instead.

### 6.2 Temporal proximity (at `store()`)

Link the new memory to the **3** most recent memories with
`created_at` within the last **600 s**, initial weight **0.2**, origin
`temporal`. Timestamp-based, so it works across process restarts. Skip
pairs already linked (reinforce instead). This encodes information
embeddings cannot: two memories laid down in the same working context.

### 6.3 Co-retrieval reinforcement (at `recall`)

For each pair among the **top 5** final results: reinforce the edge if it
exists; otherwise create it at weight **0.05**, origin `co_retrieval`.
The tiny initial weight plus decay (§6.4) is the echo-chamber counterweight:
a pair must keep co-occurring across queries to matter.

### 6.4 Decay (lazy, at read)

```
effective_weight = stored_weight · 2^(-days_since_last_reinforced / 30)
```

Computed wherever a weight is used (activation, stats); persisted only when
the row is otherwise written. Applies to all origins — `explicit` edges
start higher (0.5) rather than being exempt, so one rule covers everything.

### 6.5 Pruning + maintenance

Every **100** `store()` calls, run a synchronous, bounded maintenance
pass: delete edges with effective weight < **0.02** and expire old query
rows (§4). No background threads, no schedulers in v0.

### 6.6 Reinforcement & explicit feedback

```
reinforce (passive, co-retrieval):   w += 0.05 · (1 - w)
feedback(query_id, positive=True):   for each pair (i, j) in result_ids
                                     and each edge in path_edge_ids:
                                        w += 0.15 · e_i · e_j · (1 - w)
feedback(query_id, positive=False):  w -= 0.15 · e_i · e_j · w
```

`e_i` = stored final energy for that memory in the query row (1.0 for
fusion-only results). Updates set `last_reinforced_at` (which also resets
decay) and bump `reinforcement_count`. Feedback on an unknown `query_id`
raises `NotFoundError`; a second feedback on the same id raises
`InvalidArgumentError`. Positive feedback may also *create* edges between
result pairs that lack one (initial weight 0.05·e_i·e_j, floor 0.02,
origin `co_retrieval`).

No composite outcomes, no confidence gating, no oscillation detection: the
only high-magnitude signal is explicit, so the noise-defense machinery has
nothing to defend against.

**Amended 2026-07-25 — feedback pairs the top 5 results, not all of them.**
As originally written, "each pair (i, j) in result_ids" is C(top_k, 2): 45
edges from one feedback call at the default top_k of 10, and 4,950 at the
maximum of 100. Since `top_k` is a caller-controlled per-call argument, that is
unbounded write amplification driven by the API caller, against §6.3's
deliberately fixed 10. Feedback now pairs the top 5 results, matching §6.3;
path edges are unaffected, since they already exist and reinforcing one grows
nothing.

Measured on the `chained` benchmark profile: unbounded feedback built 943
learned edges against 150 real ones and cost two associative hits, while the
bounded form built 177 and recovered one. See the 2026-07-25 entry in
`dev/v0-dev-plan.md` for the full readings.

---

## 7. Graph confidence & vector index

### 7.1 Graph confidence

Same four components and weights as the predecessor's `graph.py` (they are
cheap and sane), computed from counts + averages, cached and invalidated on
graph writes:

| Component | Weight | 0.0 → 1.0 bands |
|---|---|---|
| node density | 0.2 | <50 → 0; 50–200 → 0.5; 200–500 → 0.8; 500+ → 1.0 (linear within bands) |
| edge density (avg edges/node) | 0.3 | <1 → 0; 1–3 → 0.5; 3–8 → 0.9; 8+ → 1.0 |
| edge quality (avg effective weight) | 0.3 | <0.35 → 0; 0.35–0.6 → 0.6; 0.6+ → 1.0 |
| reinforcement (avg reinforcement_count) | 0.2 | <2 → 0; 2–10 → 0.7; 10+ → 1.0 |

Exposed as `Stats.maturity` and `RecallResult.maturity`.

### 7.2 Vector index

An in-process `numpy` float32 matrix (ids aligned array), built lazily on
first `recall()` and updated incrementally on `store()` and `forget()`. Brute-force
cosine is < 10 ms at 50k × 384 dims — no ANN library in v0. If profiling at
larger scale demands it, `sqlite-vec` becomes an optional extra in v0.2+;
the schema already stores embeddings per-row so no migration is needed.

---

## 8. Embeddings

- Core dependencies: `pydantic>=2`, `numpy>=1.24`. Nothing else.
- `pip install synapticdb[embeddings]` adds `sentence-transformers` (which
  brings torch). Default model: `all-MiniLM-L6-v2` (384 dims), lazy-loaded
  on first use.
- `embedding_fn: Callable[[str], Sequence[float]]` always wins when given.
- If `embedding_fn is None` and the extra is not installed, the constructor
  succeeds but the first `store()` or `recall()` raises `EmbeddingError` with
  the install hint. **No silent hash-embedding fallback** — fake vectors
  produce fake retrieval and poison the graph.
- Embedding dimension is recorded in a `meta` pragma table on first write;
  a mismatched dimension later raises `EmbeddingError`.

---

## 9. Parameter budget (all of them)

| # | Parameter | Default | Provenance |
|---|---|---|---|
| 1 | `top_k` (recall) | 10 | holdout config |
| 2 | keyword/semantic candidate depth | 40 | holdout config |
| 3 | RRF `k` | 60 | holdout config |
| 4 | activation seeds | 5 | holdout config |
| 5 | activation max steps | 5 | holdout config |
| 6 | activation decay | 0.2 | holdout config |
| 7 | activation min energy | 0.05 | holdout config |
| 8 | hop bonus | 0.15/hop | holdout config |
| 9 | seed penalty | 0.2 | holdout config |
| 10 | activation blend weight (α max) | 0.45 | holdout config |
| 11 | semantic seed threshold / top-k / weight | 0.6 / 3 / 0.25 | calibrate in bench |
| 12 | temporal window / max links / weight | 600 s / 3 / 0.2 | calibrate in bench |
| 13 | co-retrieval initial weight / reinforce rate | 0.05 / 0.05 | calibrate in bench |
| 14 | explicit feedback rate | 0.15 | calibrate in bench |
| 15 | explicit `connect` weight | 0.5 | judgment |
| 16 | decay half-life / prune threshold | 30 d / 0.02 | calibrate in bench |
| 17 | maintenance interval | 100 store calls | judgment |

Constructor exposes **none** of these in v0 except `top_k` per-call. A
single private `_params` dict exists for the benchmark harness to sweep.
If real users need a knob, exposing it is a deliberate v0.x decision.

---

## 10. Benchmark harness (`bench/`, in-repo, not in wheel)

Keep the predecessor's proven protocol and baseline implementation, but use an
in-repository deterministic synthetic dataset:

- **Corpus:** 500 wholly fictional memories, four balanced content styles, and
  25 deliberate four-memory associative chains. **Queries:** 50, labeled
  `direct` (25) and `associative` (25). No personal-project content is retained.
- **Baseline:** BM25 + TF-IDF/FAISS exact cosine + cross-encoder rerank
  (`ms-marco-MiniLM-L-6-v2`), the predecessor's locked baseline config.
- **Protocol:** ingest corpus in narrative order (exercises temporal
  edges), replay a scripted usage phase (recalls + feedback) to warm the
  graph, then evaluate the holdout query set. Multiple seeds; report
  associative hits, unique wins vs baseline, direct-recall parity.
- **Integrity:** corpus link annotations and query relevance/path labels are
  evaluator-only and are never passed to a retriever. Warm-up queries are
  disjoint from the holdout queries.
- **Model fidelity:** dataset IDs are mapped to product UUIDs by the adapter;
  warm-up uses query-level positive/negative feedback; a controlled ingestion
  schedule supplies timestamps for the 600-second temporal window.
- **Gates:** (a) direct-recall within 5% of baseline; (b) ≥ 10/25
  associative unique wins, matching the original pass gate.
- **Rule:** any change to §5, §6, or §9 defaults lands with a before/after
  bench run in the PR description.

Run: `python -m bench` (deps via `pip install -e .[bench]`).

Synthetic dataset version 1 has a frozen baseline target of 25/25 direct and
10/25 associative hits. This target was measured after generating the new
corpus; its equality with the retired corpus's headline result is coincidental.

---

## 11. Package layout & budgets

```
synapticdb/
├── pyproject.toml            # name TBD (§14); deps: pydantic, numpy
├── README.md                 # 5-line quickstart that actually works
├── docs/
│   └── v0-prd.md             # this document
├── src/synapticdb/
│   ├── __init__.py           # exports: SynapticDB, Memory, RecalledMemory, RecallResult, Stats, errors
│   ├── api.py                # SynapticDB class, orchestration        (~350 lines)
│   ├── store.py              # SQLite + FTS + vector matrix          (~500 lines)
│   ├── retrieval.py          # search, RRF, blend                    (~200 lines)
│   ├── activation.py         # spreading activation                  (~150 lines)
│   ├── learning.py           # seeding, temporal, co-retrieval,
│   │                         # decay, feedback                       (~250 lines)
│   ├── confidence.py         # graph confidence                      (~80 lines)
│   ├── embeddings.py         # default model, protocol, dim check    (~80 lines)
│   └── models.py             # pydantic models + errors              (~120 lines)
├── bench/                    # harness + corpus + baseline (excluded from wheel)
└── tests/
```

Budget total: ~1,730 lines core. Hard ceiling 2,000 (principle 5).
No `_engine` facade, no satellite classes holding a back-reference to the
engine, no duck-typed graph indirection — modules take plain arguments.

### 11.1 Budget amendment (Phase 6.4, 2026-07-25)

**Measured: 2,375 core lines, excluding blanks, comments, and docstrings.
The ceiling is raised to 2,400. The per-module estimates below are retired.**

| module | estimate | measured |
|---|---:|---:|
| store.py | 500 | 1,147 |
| api.py | 350 | 450 |
| learning.py | 250 | 235 |
| activation.py | 150 | 189 |
| retrieval.py | 200 | 107 |
| models.py | 120 | 53 |
| confidence.py, embeddings.py, `__init__.py` | 160 | 179 |

The overage is one module. `store.py` measures 2.3× its estimate; every other
module is within about 40 lines of its own.

The Phase 6.4 audit looked for the duplication that would explain the gap and
did not find it. An automated scan for repeated four-line blocks returned only
function signatures and two-line idioms. Three identical row fetchers were
consolidated into `_required_row`, which recovered 10 lines. Nothing else in
`store.py` repeats at a scale worth extracting.

What the 500-line estimate omitted is scope, not slack. `store.py` owns five
tables, an FTS5 mirror kept in sync by three triggers, the vector matrix and its
incremental maintenance, edge canonicalization, lazy decay in SQL with a
capability probe and callback fallback, bounded batch pruning, and query
persistence with feedback. The estimate was written before those were specified.

**The structural constraints in section 11 still bind and still hold.** There is
no facade, no back-reference, and no duck-typed indirection. The ceiling was a
proxy for that discipline; the discipline is what the audit actually verified.

Cutting 375 lines from `store.py` now would mean deleting precondition checks
(reliability rule 5) or splitting one cohesive module across files to move lines
rather than remove them. Both trade a real property for a number.

Revisit if `store.py` passes 1,400 measured lines, which would mean the scope
grew rather than the estimate being wrong.

---

## 12. Testing requirements

| Area | Must cover |
|---|---|
| store | roundtrip, dedupe returns existing, FTS sync on insert/delete, forget cascades, dim mismatch |
| retrieval | BM25 ranking, cosine ordering, RRF overlap behavior, where-filter, empty DB |
| activation | propagation math, loop guard, min-energy pruning, hop bonus, seed penalty, max steps |
| blend | α=0 equals pure fusion; via attribution correct for all three cases |
| learning | each of the four mechanisms creates/reinforces as specced; decay math; prune removes; feedback idempotency errors; negative feedback weakens |
| confidence | band math per component; cache invalidation |
| api | quickstart script verbatim from README; error types; context manager; `:memory:` |
| bench | harness runs end-to-end on a 50-memory smoke corpus in CI (< 60 s, no model download in CI via fixture embeddings) |

Test suite budget: aim ≤ 2× core lines. The predecessor's 9,500-line suite
guarded contracts nobody consumed.

---

## 13. Performance targets

Measured at 10,000 memories, 384-dim embeddings, M-series laptop.

| Operation | p99 |
|---|---|
| `store()` (excluding embedding call) | < 50 ms |
| `recall` full pipeline (excluding embedding call) | < 100 ms |
| `feedback` | < 20 ms |
| maintenance pass | < 200 ms, synchronous, every 100 store calls |
| `stats` | < 10 ms |

---

## 14. Package decisions

1. **Distribution and import name:** `synapticdb`.
2. **Version scheme:** start at `0.1.0` as a new package rather than
   continuing the predecessor's `0.4.x`.
3. **License:** MIT.

## 15. Roadmap after v0 (summary; each phase gets its own doc)

- **v0.2 — learning quality:** recency-weighted ranking, opt-in
  `memory_in_response` implicit signal (embedding similarity of agent
  response, the one low-noise implicit signal), decay/lifecycle tuning —
  all benchmark-gated.
- **v0.3 — agent integration:** MCP server (`synaptic serve`), minimal CLI
  (`stats`, `doctor`), async variants if server usage demands them.
- **v0.4 — advanced memory, benchmark-gated:** short/long-term tiers +
  graduation only if the flat graph measurably degrades at scale;
  type classification; cluster consolidation.
- **v0.5 — scale-out, demand-driven:** `namespace` column, Postgres
  backend, opt-in anonymized telemetry export (the flywheel, revisited only
  with real users).
