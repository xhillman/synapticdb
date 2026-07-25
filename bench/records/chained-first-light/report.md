# SynapticDB Benchmark

- Decision: **FAIL**
- Dataset: `963fedf14b723814e7f98d5189d613edbf54cfafc7aa7a8fb713c72eceab2779`
- Baseline: `baseline`
- Candidate: `synaptic`
- Top-k: `10`
- Runtime: Python `3.11.4`

| Seed | Baseline Direct | Candidate Direct | Baseline Assoc | Candidate Assoc | Unique Wins | Reproduced | Direct Parity | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 25/25 | 25/25 | 10/25 | 17/25 | 7/25 | Y | Y | FAIL |

Gate: direct recall within `5%` of baseline and at least `10` path-backed associative unique wins on every seed.

| Seed | Measurement | Before | After | Series | Gate | Claim |
|---:|---|---:|---:|---|---|---|
| 1337 | trajectory | 17 | 17 | — | PASS | using the system improves it: warm associative hits >= cold |
| 1337 | diversity | 193 | 189 | 193 → 192 → 188 → 189 | FAIL | learning does not collapse the result set: repeat distinct >= first |
Baseline reproduction target: `25/25` direct and `10/25` associative hits.
