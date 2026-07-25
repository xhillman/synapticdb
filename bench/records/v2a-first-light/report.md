# SynapticDB Benchmark

- Decision: **FAIL**
- Dataset: `2e1263f9557e6465bcb53eefd58a550c728e93ea96ef52d3525428e3125d3cd8`
- Baseline: `baseline`
- Candidate: `synaptic`
- Top-k: `10`
- Runtime: Python `3.11.4`

| Seed | Baseline Direct | Candidate Direct | Baseline Assoc | Candidate Assoc | Unique Wins | Reproduced | Direct Parity | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 25/25 | 25/25 | 10/25 | 17/25 | 7/25 | Y | Y | FAIL |

Gate: direct recall within `5%` of baseline and at least `10` path-backed associative unique wins on every seed.

| Seed | Measurement | Before | After | Gate | Claim |
|---:|---|---:|---:|---|---|
| 1337 | trajectory | 17 | 17 | PASS | using the system improves it: warm associative hits >= cold |
| 1337 | diversity | 195 | 194 | FAIL | learning does not collapse the result set: repeat distinct >= first |
| 1337 | decay_direct_recall | 25 | 25 | PASS | an aged graph still serves search: aged direct >= warm |
Baseline reproduction target: `25/25` direct and `10/25` associative hits.
