# nd-takehome

Take-home: bootstrapping a natural-deduction prover past its training length

Forked from https://github.com/chainik1125/nd-takehome (`upstream` remote). See `README.md` and
`spec.md` for the task itself.

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

I haven't thought about logic in a long time, define stuff more often than you'd think.

## Data naming and canonicalization

- `data/heldout.jsonl` is our own <= 6-line split (the README calls this the "held-out set").
  Never call it "val": "validation" means the provided `targets/validation_36.jsonl` only.
- Atoms are canonical: renamed so they first appear in the prompt as P, Q, R, S. The map is fixed
  by the prompt alone. Rename on formula tuples or prompt/compact tokens (`ndtok.canonical_map`,
  `rename_formula`, `rename_tokens`), never on spec text, where the rule `R` is spelled like the
  atom `R`. `prove.py` must canonicalize the incoming prompt and invert the map on the output.
- Held-out is frozen: never remove or regenerate its records. New data classes are added with
  `make_data.py --append`, which puts 5% of the new theorems into held-out (so the class is
  measurable in-distribution) and the rest into train. Records carry `round` (1 = original split).
- Theorem identity is `key` (atom renaming + premise order folded, see `gen.theorem_key`). Every
  record carries it. `gen.sample` dedupes on it at generation time, so a dataset never contains two
  records with the same key; `make_data.py` splits on it. Never dedupe on `thm` or `prompt`.

## Notes

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `.claude/notes/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `.claude/notes/local/*.md`.

See `~/.claude/CLAUDE.md` for the full convention.

Current notes:
<!-- As notes are added under .claude/notes/, list them here, one per line: -->
<!-- - [Title — when to read](.claude/notes/foo.md) — short gloss -->
