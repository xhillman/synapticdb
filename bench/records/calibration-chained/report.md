# SynapticDB Benchmark

- Decision: **FAIL**
- Dataset: `6e95a1cbd637fd8f78e3762fd35d33a5bbc6b08da4e28a560cea9fa2402aeca9`
- Baseline: `baseline`
- Candidate: `synaptic`
- Top-k: `10`
- Runtime: Python `3.11.4`

| Seed | Baseline Direct | Candidate Direct | Baseline Assoc | Candidate Assoc | Unique Wins | Reproduced | Direct Parity | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 25/25 | 25/25 | 10/25 | 17/25 | 7/25 | Y | Y | FAIL |

Gate: direct recall within `5%` of baseline and at least `10` path-backed associative unique wins on every seed.

| Seed | MRR | MRR floor | Chain coverage | Separation | Confidence AUC | Gate |
|---:|---:|---|---:|---|---:|---|
| 1337 | 0.6071 | — | 0.5600 | +0.3823 | 0.9940 | PASS |

Confidence AUC must reach `0.80`: a correct answer must outscore a question with no answer at least that often, or the field cannot be thresholded. MRR must not regress against `--compare-to`. Chain coverage is reported, not gated: we cannot yet argue which direction is good.
Baseline reproduction target: `25/25` direct and `10/25` associative hits.
