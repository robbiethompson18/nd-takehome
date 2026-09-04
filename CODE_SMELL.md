# Code smells

## 2026-09-03 — `harvest.py` prints "FROZEN-MODEL CONTROL" for every harvest

`logs/pod_loop.log` contains six lines reading `FROZEN-MODEL CONTROL: N/930 distinct theorems
solved`, but only three of them are the control. The other three are the loop harvesting with its
*current* (retrained) model — `harvest.py` prints the same banner either way. Reading the log
top-to-bottom, the control appears to go 65 → 75 → 115 → 66 → 134 → 64, which is nonsense.

The real split, by the checkpoint named on the preceding line:

- loop model: 65 (r1, still the pretrain model) → 115 (r2) → 134 (r3)
- frozen control: 75 / 66 / 64 per round, cumulative union 75 → 79 → 82

Fix: take the banner text from the checkpoint being harvested, or pass a `--label`. Until then the
log is a trap for anyone (including a grader) reading it without the surrounding context.

## 2026-09-03 — the frozen control's cumulative trajectory is not printed anywhere

`writeup.md` reports the control as 75 → 79 → 82. That is the cumulative union of
`data/harvest/ctrl_r{1,2,3}.jsonl` on `key`, computed by hand after the fact; the loop only ever
prints the per-round counts. Worth emitting it in `harvest.py` so the headline control number is
reproducible from a log rather than from a one-off snippet.
