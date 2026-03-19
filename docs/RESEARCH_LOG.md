# Research Log
(See docs/CONTEXT.md)

## Entry Format

### Date: YYYY-MM-DD

#### Goal
What are you testing?

#### Change
What did you modify?

#### Result
What happened?

#### Insight
What did you learn?

#### Next Step
What will you try next?

---

## Example

### Date: 2026-03-19

#### Goal
Investigate fitness gap between qd_20260318_201102 (0.861) and better runs (0.90+).

#### Change
Audited run metadata; no config.json found for any run.

#### Result
Identified: missing --seed, possible init/generations config drift.

#### Insight
Runs without --seed are non-reproducible; config snapshot critical.

#### Next Step
Add --seed by default; write config.json + git hash to every run.
