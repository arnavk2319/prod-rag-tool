# Evals

- `golden/` approved cases (human-promoted, protected). `golden_proposed/` agent drafts.
- `thresholds.json` approved (protected). `thresholds.proposed.json` agent proposal.
- `redteam/` adversarial cases. `results/` outputs; `results/baseline.json` is the last released run.
- Suites: `smoke` (per PR, cheap, cached) and `full` (nightly, release candidate).
- A change that lowers any metric below its threshold fails CI. Do not edit thresholds to pass.
