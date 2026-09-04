# Stage-1 frozen checkpoints

Read-only copies of the pre-RL models. Every Stage-2 claim is measured against these; nothing
may overwrite them. `ckpts/stage1_*.pt` are the working copies and may be replaced.

Note: `.gitignore` excludes `*.pt`, so these are NOT in git. The exam requires shipping the
Stage-1 and final checkpoints with the submission, so that exclusion has to be revisited
before we hand anything over (12 MB each, so committing them is fine).

## Provenance

| | |
|---|---|
| architecture | 4 layers, d=256, 4 heads, MLP 4x, pre-norm RMSNorm, no biases, untied head |
| params | 3,164,928 (3.17M) |
| positional | `stage1_nope.pt` = none (NoPE); `stage1_rope.pt` = rotary, base 10000 |
| vocab | 34 tokens (`ndtok.VOCAB`), compact format: no line indices, no PR lines, content-addressed citations |
| training data | `data/train.jsonl`, 142,118 cap-6 theorems, deduped by `gen.theorem_key` |
| held-out | `data/heldout.jsonl`, 2,000 theorems (400 each at lengths 2-6), disjoint by key |
| hyper-params | batch 1024, 6000 steps (43 epochs), AdamW lr 1e-3 cosine to 1e-4, warmup 200, wd 0.1, clip 1.0, bf16 |
| hardware | 1x H100 SXM (RunPod), 4.6 min (NoPE) / 6.2 min (RoPE) |
| decoding | greedy everywhere below |

## Numbers (all verifier-judged, proof must prove the prompted sequent)

| set | n | NoPE | RoPE |
|---|---|---|---|
| our held-out, lengths 2-6 | 2000 | 99.55% [99.15, 99.76] | 99.75% [99.42, 99.89] |
| `targets/validation_36` overall | 36 | 19.4% | 19.4% |
| `targets/validation_36` bin `<=6` | 12 | 58.3% | 58.3% |
| `targets/validation_36` bin `>6` | 24 | 0% | 0% |
| `data/long/rl_targets` (gen_lines 7-16) | 930 | 4.52% [3.36, 6.05] | 3.98% [2.90, 5.44] |
| `data/long/transfer` (gen_lines 7-16) | 931 | 4.94% [3.72, 6.53] | 4.73% [3.54, 6.29] |

Verified proofs by the length actually written (both long pools combined):

    NoPE {4:4, 5:14, 6:18, 7:49, 8:3}      robust frontier (>=5 distinct verified) = 7
    RoPE {4:2, 5:12, 6:22, 7:41, 8:4}      robust frontier = 7

**P = 7.** NoPE and RoPE are indistinguishable at Stage 1; the positional scheme has not
separated them yet.

## Failure mode

~75% of failures on the long pools are `bad line cite`, which is the verifier's name for a
citation `ndtok.decode` could not resolve (it fills those with the line's own index). Inspecting
them: the model omits the sub-derivation of a minor premise and jumps to the conclusion --
`N4 P : IMPE N3 N4` where `( Q & R )` was never derived. Concentrated on IMPE (940) and NEGE
(212), at lines 4-5, in proofs that end by line 6. It is proof *compression* to fit the trained
length, not a formatting bug.
