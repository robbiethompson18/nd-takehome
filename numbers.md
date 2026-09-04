# numbers.md

Every number in `writeup.md`, and the file it came from. Commands are run from the repo root with
`uv run python ...`. Where a number is a solve rate, "source" is the prove.py output that was
scored plus the pool it was scored against; `score_pool.py` prints the rate, the Wilson interval,
the by-length breakdown and the robust frontier in one go.

Charts and their CSVs live in `charts/` here and in
`personal-website/posts/natural-deduction-takehome/charts/` (same files; `extract.py` there is what
builds the CSVs out of this repo).

## Dataset

| number | value | source |
| --- | --- | --- |
| Train theorems | 142,118 | `data/train.jsonl` (`wc -l`), `data/meta.json` |
| Held-out theorems | 2,000 (400 per length 2-6) | `data/heldout.jsonl`, `data/meta.json` |
| Train by length | 7,169 / 16,813 / 39,383 / 39,269 / 39,484 for lengths 2-6 | `data_report.py data/train.jsonl` |
| Trivial share of train | 8.4% at length 2, ≤0.1% at 3-6, 0.5% overall | `data_report.py data/train.jsonl` |
| Median body tokens | 30 / 38 / 38 / 47 / 50 for lengths 2-6 | `data_report.py data/train.jsonl` |
| Held-out theorems that are a renaming or premise-reordering of a training theorem | 0, by construction | dedupe is on `key` = `gen.theorem_key`; asserted in `make_data.py` before writing |
| Sequents from `targets/` dropped from both splits | 100 | `make_data.py` stdout, `data/meta.json` → `target_overlap_dropped` |
| Dead lines in generated data | 0 of 37,748 lines (cap-6 sample); 0 of 20,776 (7-16 pools) | `prune.py data/gen_2_6.jsonl` |
| Share of training proofs using each rule | IMPE 56.8%, ANDE1 32.7%, ANDE2 32.2%, NEGE 19.8%, BOTE 19.1%, ANDI 18.8%, ORI1 16.5%, ORI2 16.4%, DN 14.2%, AS 8.6%, IMPI 7.8%, REIT 1.3%, ORE 0.6%, NEGI 0.2% | `charts/rule_use.csv` (chart 1) |
| Boxed share of training proofs | 0.1% / 0.2% / 3.6% / 10.0% / 12.4% at lengths 2-6 | `data/train.jsonl`, field `max_depth > 0` |
| Boxed share of the 7-16 pools | 94.8% RL targets, 92.9% transfer; depth ≥2 24.2% / 22.7% | `make_eval_pool.py` stdout, `data/long/meta.json` |
| RL-target / transfer pool sizes | 930 / 931, ~100 per generating length 7-15, 30/31 at 16 | `data/long/rl_targets.jsonl`, `data/long/transfer.jsonl` |

## Model and training

| number | value | source |
| --- | --- | --- |
| Parameters | 3.17M (4 layers, d=256, 4 heads, no biases, untied head) | `model.py` `Model.n_params()`; printed at the top of `logs/stage1_*.log` |
| Vocabulary | 34 tokens, all distinct | `ndtok.VOCAB`; asserted in `ndtok.py` and `test_ndtok.py` |
| Steps / batch / epochs | 6,000 steps at batch 1,024 = 43.2 epochs over 142,118 theorems | `logs/stage1_nope.log`, `logs/stage1_rope.log` |
| Step time | 45 ms (NoPE), 61 ms (RoPE) on one H100 SXM | `logs/stage1_nope.log`, `logs/stage1_rope.log` |
| Wall clock | 4.6 min (NoPE), 6.2 min (RoPE) | last line of each `logs/stage1_*.log` |
| Final train / held-out loss | 0.0017 / 0.0020 (NoPE), 0.0012 / 0.0026 (RoPE) | `charts/loss_curves.csv` (chart 2) |
| GPU spend | <$10 total, H100 SXM at $3.49/hr | `logs/pods.json`, `pod.py` cost lines in `logs/pod_*.log` |

## Stage 1: held-out (≤6 lines)

| number | value | source |
| --- | --- | --- |
| Held-out greedy solve rate, NoPE | 1,991/2,000 = 99.55% [99.15, 99.76] | `ckpts/stage1_nope.pt` → `extra.final`; `logs/stage1_nope.log` |
| Held-out greedy solve rate, RoPE | 1,995/2,000 = 99.75% [99.42, 99.89] | `ckpts/stage1_rope.pt` → `extra.final`; `logs/stage1_rope.log` |
| By length, NoPE | 400/400, 398/400, 399/400, 398/400, 396/400 for lengths 2-6 | same |
| By length, RoPE | 399/400, 400/400, 400/400, 400/400, 396/400 | same |
| Proofs shorter than the reference | 42 (NoPE), 43 (RoPE) of 2,000 | `prove.py --ckpt ckpts/stage1_nope.pt --in data/heldout.jsonl`, scored against `data/heldout.jsonl` |
| Held-out solve rate over training | ~97% at epoch 7, ~99.5% from epoch 15 | `charts/solve_curve.csv` (chart 3) |

Caveat carried in charts 2 and 3: the mid-training evaluations in these two runs read `val[:1024]`
of a length-ordered file, so those curves cover lengths 2-4 only and read slightly high. The final
figures above are the full 2,000.

## Stage 1: past the cap (7-16 line pools)

Pretrain-only = `ckpts/stage1_nope.pt`, greedy, one attempt per theorem.

| number | value | source |
| --- | --- | --- |
| RL targets, pretrain-only | 42/930 = 4.5% [3.4, 6.0] | `score_pool.py --proofs logs/p_pretrain_rl_targets.jsonl --pool data/long/rl_targets.jsonl` |
| Transfer, pretrain-only | 46/931 = 4.9% [3.7, 6.5] | `score_pool.py --proofs logs/p_pretrain_transfer.jsonl --pool data/long/transfer.jsonl` |
| Solve rate by proof length | 25% at 7, 2% at 8, 0 from 9 on (filed under `min(generating, written)`) | `charts/solve_by_length.csv` (writeup's solve-rate chart) |
| Longest proof written on the pools | 8 lines | `charts/written_length.csv` |
| Robust frontier, pretrain-only | 7 lines (49 verified 7-line proofs NoPE, 41 RoPE; only 3 and 4 at length 8, under the ≥5 bar) | `charts/written_length.csv`, `charts/frontier.png`; `score_pool.py` prints it directly |
| Verifier rejections by reason | bad line cite 2,818 of 3,553 (79%); then final-formula-not-conclusion 208, parse error 198, rule check failed 153, bad box cite 149, ends-inside-subproof 27 | `charts/error_breakdown.csv` (chart 4); NoPE + RoPE pooled over both pools |
| Solves whose written length equals the generating length | 80% at label 7, 15% at label 8, 0% at label 9 and beyond | `charts/solved_lengths.csv` (chart 5) |
| NoPE vs RoPE, held-out | 99.55% vs 99.75%, a 0.2pp gap against ~0.15pp SE | the two `extra.final` blocks above |
| NoPE vs RoPE, pools | 4.5% vs 4.0% (RL targets), 4.9% vs 4.7% (transfer); frontier 7 for both | `score_pool.py` on the corresponding prove.py outputs |

## Stage 2: expert iteration

Three rounds, NoPE only. Per-length tables in the writeup come from the `by gen_lines` line of
`score_pool.py` on each file below.

| number | value | source |
| --- | --- | --- |
| RL targets after 3 rounds | 135/930 = 14.5% [12.4, 16.9] | `score_pool.py --proofs logs/p_r3_rl_targets.jsonl --pool data/long/rl_targets.jsonl` |
| Transfer after 3 rounds | 97/931 = 10.4% [8.6, 12.5] | `score_pool.py --proofs logs/p_r3_transfer.jsonl --pool data/long/transfer.jsonl` |
| Transfer gain | 4.9% → 10.4%, +5.5pp against ~1.3pp SE | the two rows above |
| Harvest budget per round | k=32 at T=1.0 over 930 RL targets = 29,760 attempts | `logs/pod_loop.log` |
| Retrained model, cumulative distinct theorems harvested by round | 65 → 115 → 134 | `logs/pod_loop.log` ("merged N files -> … over X theorems"); `data/harvest/r{1,2,3}.jsonl` |
| Frozen-model control, same attempt budget, no retraining | per round 75 / 66 / 64; cumulative 75 → 79 → 82 (+7) | `data/harvest/ctrl_r{1,2,3}.jsonl`, cumulative union on `key` |
| Verified proofs at length 7 / 8, by round | RL targets 23/2 → 54/8 → 68/10 → 75/16; transfer 24/1 → 41/4 → 45/5 → 48/9 | `charts/harvest_loop.csv` (harvest-loop chart) |
| Robust frontier after expert iteration | 8 lines on both pools (round 1 clears it on RL targets, 4 distinct L8 on transfer; round 2 clears both) | `score_pool.py` frontier line on each round's files |
| **L − P** | **8 − 7 = 1** | the two frontier rows above |
| Nine-line proofs written | 0, in every round, on both pools | `verified, by length PRUNED` line of `score_pool.py` on every round |
| Ablation: harvest fraction of batch | f=0.1 / 0.25 / 0.5 | `ckpts/stage2_nope_f*.pt`, `logs/p_f*_*.jsonl` |
| Ablation: fine-tune without harvested proofs | 35/930 = 3.8% [2.7, 5.2] on RL targets | `score_pool.py --proofs logs/p_noharv_rl_targets.jsonl --pool data/long/rl_targets.jsonl` |

## Numbers deliberately not reported

- **Solve rate by "true" difficulty.** We have no shortest-proof labels; `gen_lines` is the
  generator's proof length, an upper bound that is honest to about length 8 and fiction beyond.
  Getting real ones needs a bounded prover, which we did not write. Every per-length rate past 8 in
  this file should be read as a property of the pool, not of the model.
- **`targets/validation_36.jsonl` and the test sets.** Scored once early (7/36, 0/24 on the `>6`
  bin, both models) and then left alone; not tuned against, and not part of any claim above.
