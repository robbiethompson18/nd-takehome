# ND take-home: submission

Fork of [chainik1125/nd-takehome](https://github.com/chainik1125/nd-takehome). The original
assignment text follows below the divider. This section is how to reproduce every number.

## What is here

| path | what |
|---|---|
| `prove.py` | the required sampling interface (same arguments and output schema as `submission_template/prove.py`; no verifier in the loop) |
| `requirements.txt` | `torch`, `numpy`. Nothing else is needed to run `prove.py` |
| `ckpts/stage1_nope.pt`, `ckpts/stage1_rope.pt` | Stage-1 models (3.17M params, NoPE / RoPE). `ckpts/frozen/` holds read-only copies; `.gitignore` drops those, they are byte-identical to the two above |
| `ckpts/stage2_*.pt` | Stage-2 (expert iteration) models. `stage2_r3.pt` is the final model |
| `data/train.jsonl`, `data/heldout.jsonl` | cap-6 training set and our held-out split, disjoint by theorem (`key`) |
| `data/long/rl_targets.jsonl`, `data/long/transfer.jsonl` | 7-16 line pools: what RL samples against, and what it never sees |
| `data/harvest/` | verified proofs the model wrote during expert iteration, per round |
| `logs/` | every training log and every `prove.py` output the numbers come from |
| `charts/` | figures for the write-up |
| `ndtok.py` | model-facing proof format (no line indices, no PR lines, content-addressed citations) and the decoder back to `spec.md` |
| `gen.py` | backward proof generator; `make_data.py` builds the split; `make_eval_pool.py` builds the 7-16 pools |
| `prune.py` | dead-line pruner; every reported length is given written and pruned |
| `train.py`, `model.py`, `harvest.py`, `merge_harvest.py`, `score_pool.py`, `pass_at_k.py` | training, sampling, scoring |
| `pod.py` | RunPod driver used for the GPU runs (needs `RUNPOD_API_KEY`) |

Terminology follows the assignment: **held-out** is our own <= 6 split, **validation** is
`targets/validation_36`, **test** is `targets/test_*`, **RL targets** and **transfer** are our two
7-16 line pools.

## Setup

```bash
pip install -r requirements.txt        # or: uv sync
python test_ndtok.py                   # round-trips all 57 provided proofs through the tokenizer
```

## Run the submitted model

```bash
python prove.py --ckpt ckpts/stage2_r3.pt --in targets/validation_36.jsonl --out out.jsonl --greedy
python eval_targets.py --proofs out.jsonl
python prove.py --ckpt ckpts/stage2_r3.pt --in targets/test_short_prompts.jsonl --out test_short.jsonl --greedy
python score_test.py test_short.jsonl
```

36 theorems take ~2 s on an M4 Pro (CPU/MPS); 1,000 theorems take well under a minute on any GPU.

## Reproduce

All seeds are fixed. Hardware for the GPU steps: one H100 SXM on RunPod. CPU steps ran on a laptop.

**Stage 1 data** (~15 min CPU for the split, ~3 min for the nested round, ~5 min for the pools):

```bash
python make_data.py --per-len 40000 --val-per-len 400 --seed 0 --seconds 900
python make_data.py --append --nested --per-len 1500 --lo 4 --hi 6 --seed 7 --seconds 150
python make_eval_pool.py --per-len 200 --seed 1 --seconds 300
python data_report.py data/train.jsonl data/heldout.jsonl      # distributions in the write-up
python prune.py data/gen_2_6.jsonl                             # 0 dead lines in generated proofs
```

The split on disk additionally had 669 premise-reorderings and 48 unused-premise records removed
in place after `key` and the unused-premise check were added to the generator; a fresh run
produces neither. `data/meta.json` records all of it.

**Stage 1 training** (4.6 min NoPE, 6.2 min RoPE on the H100; ~45 ms/step at batch 1024):

```bash
python train.py --pos nope --batch 1024 --steps 6000 --lr 1e-3 --seed 0 --out ckpts/stage1_nope.pt
python train.py --pos rope --batch 1024 --steps 6000 --lr 1e-3 --seed 0 --out ckpts/stage1_rope.pt
```

Held-out greedy solve rate by length is printed at the end (`logs/stage1_*.log`, "FINAL").

**Stage 2, frozen-model control and harvest** (k = 32 samples per theorem at T = 1.0):

```bash
python harvest.py --ckpt ckpts/stage1_nope.pt --in data/long/rl_targets.jsonl --k 32 --temperature 1.0 --seed 0 --out data/harvest/nope.jsonl
python pass_at_k.py --ckpt ckpts/stage1_nope.pt --in data/long/transfer.jsonl --k 32 --out logs/passk_stage1_transfer.json
```

`harvest.py` prints the frozen-model control: distinct theorems solved at this budget with no
retraining. Only `rl_targets` is ever harvested; the script refuses the transfer, held-out,
validation and test files.

**Stage 2, expert iteration.** Each round fine-tunes from the frozen Stage-1 model with the
harvested proofs mixed into half of every batch, then re-harvests with the new model
(~20 s of training per round; harvesting dominates):

```bash
python train.py --init ckpts/stage1_nope.pt --extra data/harvest/all.jsonl --extra-frac 0.5 \
    --lr 1e-4 --steps 400 --batch 512 --warmup 20 --out ckpts/stage2_r1.pt
python harvest.py --ckpt ckpts/stage2_r1.pt --in data/long/rl_targets.jsonl --k 32 --temperature 1.0 --out data/harvest/r1.jsonl
python merge_harvest.py data/harvest/nope.jsonl data/harvest/rope.jsonl data/harvest/r*.jsonl --out data/harvest/all.jsonl
# repeat for r2, r3
```

Control: the same fine-tune with no harvested data (`--extra` omitted) is `ckpts/stage2_nope_noharvest.pt`.
Mix-fraction sweep: `--extra-frac 0.1 / 0.25 / 0.5` on the round-0 harvest are `ckpts/stage2_nope_f*.pt`.

**Scoring any model on any pool:**

```bash
python prove.py --ckpt ckpts/stage2_r3.pt --in data/long/transfer.jsonl --out logs/p_r3_transfer.jsonl --greedy --max-new 400
python score_pool.py --proofs logs/p_r3_transfer.jsonl --pool data/long/transfer.jsonl
```

`score_pool.py` reports solve rate with a Wilson interval, the breakdown by generating length, the
histogram of lengths actually written and the same after pruning dead lines, and the robust
frontier (longest length with >= 5 distinct verified proofs) on both. Every `logs/p_*.jsonl` is a
`prove.py` output that can be rescored this way.

## Notes for the reader

- Generating length (`gen_lines`) is an upper bound on a theorem's shortest proof, not a difficulty.
  Lengths we report for model output are lengths actually written, with pruned lengths alongside.
- The tokenizer omits line numbers and PR lines from the model's vocabulary. Citations are resolved
  by `ndtok.decode` as an exact-match lookup on the model's own output, with no search and no
  verifier call. See the docstring at the top of `ndtok.py`.
- Test set, scored once each with greedy decoding (outputs in `logs/p_stage1_nope_test_*.jsonl` and
  `logs/p_stage2_r3_test_*.jsonl`, rescore with `python score_test.py <file>`): Stage-1 45.3% short /
  4.5% long; final (round 3) 47.6% short / 4.3% long.

---

# Take-home: bootstrapping a natural-deduction prover past its training length


## TL;DR: Spend 2-4 hours trying to increase the complexity of theorems generated by models with a fixed pre-training capability through RL.
You are trying to maximize the number: L-P

L: The maximum length (number of proof steps) of theorems proved by an RL trained model at a given accuracy
P: The maximum length (number of proof steps) of theorems proved by the pre-trained model at a given accuracy

Writeup your results and email me the Google doc!

## Introduction

Reinforcement learning is the most important technique which has allowed models to go beyond their training data to generate new knowledge. Your goal in this take-home is to try to use the natural deduction setting to establish a baseline for new knowledge generation. There are three parts to the take home:

1. Phase 1: Generate a pre-training dataset of sound natural deduction theorems, on proofs of a _fixed number of inference steps_ (i.e. proof lines — one rule application per line) and pre-train a GPT-2 style transformer on it.

    - It is worth spending some time thinking about how you're going to (ask Claude to) tokenize the proofs and what stumbling blocks that choice might create.

2. Phase 2: Use your favourite RL technique in order to get the highest accuracy rate you can on a _held out set_ that I provide in this repo for as many inference steps beyond the maximum number in the pre-training data. This is the simplest version of novelty (longer proofs than those the models was trained on) which is easily quantifiable - a proof is more novel insofar as the gap between pre-training and post-training proof steps increases.

    - There is a validation set of hand-curated problems that you can use here: `targets/validation_36.jsonl` (reference proofs in `targets/validation_36_reference_proofs.jsonl`)

    - When you submit your final version, run your candidate algorithm **once** on the test set here — `targets/test_short_prompts.jsonl` and `targets/test_long_prompts.jsonl`, scored with `python score_test.py <output>` — and report those numbers in the writeup TL;DR.

    - Think creatively! A decent attempt at this problem executes a known and appropriate RL algorithm well. But there are many creative approaches one could take: increasing the pre-training quality, changing architectural points, combining SFT with RL, trying to get curriculum learning to work etc... I would focus on getting a working version of something reasonable first and the iterate from there.

3. Phase 3: Writeup your results in a Google doc and submit a git link to your fork of this repo so I can ask Claude about your code.

    - Do not neglect this part! I will assess the communication of part 3 as around 40% of the total value of the take home. 

    - This is particularly difficult to do well when Claude has generated most of the code and more of the research ideas than you might care to admit. Being able to keep track of the research whilst delegating effectively to Claude and communicating that well to others is a crucial skill for this project. 

    - Focus on figures in the writeup. The best writeups will have 2-5 single sentence bullet points with accompanying figures, followed by sections on each of those figures. 

    - Be careful with how you use Claude to help you with writing. It is usually very bad.



## Advice

1. Move quickly: This project is really only possible within 2-4 hours if you use Claude very aggressively.

2. Focus on the core metrics: The take-home has a fairly well defined number that we're trying to optimize. Claude will get sidetracked and either over-focus on it to the point of trivializing or cheating the result, or under-focus on it and pivot to something marginal. The key skill I'm trying to test is how comfortable you are walking this tightrope.

3. Think of the write-up in your head as you go: Ask yourself every 30m if the current thread is moving you forward, and what that would contribute to the writeup.

4. The time limit is there out of respect for your time. If you want an extra ten minutes to writeup, that's no problem. 

## Provided infrastructure and tools

To make life a little bit easier, I've provided a few tools that you can use for the experiment. 


| path | what |
|---|---|
| `spec.md` | the logic, the proof token format, the rule table — **read first** |
| `nd_verify/` | the verifier (`verify_text`). Ground truth for everything. |
| `verify_cli.py` | judge a jsonl of attempts; prints solve rate and failure reasons |
| `eval_targets.py` | score a `prove.py` output against the validation set (or any target file of yours): solve rate by bin and by length with CIs, written vs reference proof length, failure reasons |
| `try_proof.py` | write a proof by hand for any of the 36 theorems (or any sequent) and get the verifier's verdict — the fastest way to learn the format |
| `examples/proofs_2_to_8.txt` | 21 verifier-accepted proofs, three per length 2–8 (format illustration only) |
| `targets/validation_36.jsonl` | **validation set**: 36 classic sequents (Halbach-style) with a length bin — inspect freely |
| `targets/validation_36_reference_proofs.jsonl` | the same 36 with one verified reference proof each, sorted by length (2–18 lines) — to see what a good proof at each length looks like and to check your own proofs' lengths against. **Never train on these or on renamings of them.** |
| `targets/test_short_prompts.jsonl`, `targets/test_long_prompts.jsonl` | **test set**: 267 + 532 anonymised prompts (textbook problems from Carr's *Natural Deduction Pack* mixed with synthetic theorems); short = provable in ≤6 lines, long = needs more. You only ever see the aggregate score. |
| `score_test.py` | scores a `prove.py` output on a test file and prints **only** "x% passed" |
| `submission_template/prove.py` | the sampling interface we will run on private targets |

Nothing else is provided. You write the proof generator, the tokeniser/model, the training and RL code, and the evaluation harness.

## Phase 1: Generating a proof dataset



## 1. The fixed setting

- **Logic and format:** exactly as in `spec.md`. A proof's *length* is its
  number of lines (premise lines included). The verifier reports it.
- **Cap:** `L = 6`. Your supervised training data may contain only proofs of
  length ≤ 6. Everything the model is trained on before RL must satisfy this
  (we will check the dataset you submit).
- **Model:** any decoder-only transformer, **trained from scratch** on your own
  generated data. No pretrained weights, no proofs written by humans or by
  other models, no distillation from an LLM. Any size is allowed; we suggest
  ≤ 10M parameters (the reference is 4 layers, d=256, 3.3M) — bigger is not
  the point, and it makes Stage 2 comparisons less interpretable.
- **Verifier:** `nd_verify` is the only judge. Report solve rates as
  *verifier-accepted proof of exactly the prompted sequent*. Never modify it;
  if you think it is wrong, show a minimal example in the write-up.
- **Hardware.** Measured with the reference 4-layer, d=256 model (3.3M
  params) on 100k cap-6 proofs at batch 128: **T4 0.32 s/step** (2k steps ≈
  11 min, 6k ≈ 32 min), **L4 0.045 s/step** (6k ≈ 5 min), **8-core CPU
  2.5 s/step** (2k ≈ 1.4 h; ~0.5 s/step at batch 32, ~0.24 s/step for a
  2-layer d=128 model). About 2k steps at batch 128 already gives a usable
  Stage-1 model (val loss ≈ 0.03); 6k is comfortable. So Stage 1 is fine on
  CPU; Stage 2 (many sampled attempts + retraining per round) is where a small
  GPU pays off — a free-tier T4 is plenty. No GPU → ask us for a RunPod.
- **Decoding:** report greedy decoding as the baseline number everywhere. If
  you evaluate any other way, state exactly what you did (temperature, seeds,
  number of runs) so it can be reproduced. Always say n and give a confidence
  interval (Wilson is fine).

## 2. Stage 1 — data and supervised training

Build a generator of **valid** proofs (every proof you emit must pass
`verify_text`), produce a training set of proofs with length 2–6, hold out an
evaluation set, and train a model that proves held-out theorems.

Deliver:
1. The generator, with a short description of how it samples proofs and what
   its length / rule / premise distributions look like (a histogram or two).
2. A train / held-out split that is **disjoint by theorem** (same sequent must
   not appear on both sides), plus your estimate of how much of the held-out
   set is a mere atom-renaming of a training theorem.
3. Held-out solve rate (greedy), **broken down by proof length 2..6**, with
   CIs. Reference point: a well-trained small model reaches ≥ 85% greedy in
   this regime.

Things a careful person notices here: what fraction of your theorems are
trivial (conclusion is a premise, `F ⊢ …`, `A ⊢ A v A`, …) and whether that
inflates the number; whether "length 6" targets are harder than "length 3"
targets *within* the training range; what the model's failures look like
(`verify_cli.py --reasons`).

## 3. Stage 2 — RL beyond the cap

Starting from your Stage-1 model, use the verifier as the reward and try to
make the model prove theorems that need **more than 6 lines**. Reward means:
the verifier accepts the emitted proof **of the prompted sequent** (matching
premises, final line = the conclusion). A valid proof of some other theorem
earns nothing for that target — though you are free to relabel such
by-products and reuse them as training data, if you say so and keep them
disjoint from every evaluation set. Any method:
expert iteration / rejection-sampling fine-tuning, hindsight relabelling of
partial proofs, GRPO/PPO-style on-policy RL, curriculum over length, search
at inference time, or something of your own. We care about how far you get
*and* how well you know how far you got.

Set-up you must include:
- **Targets:** a pool of theorems whose generating proofs are 7–16 lines,
  disjoint from Stage-1 training and from each other's roles below. The
  theorems RL samples against are the *RL targets*; a further pool RL never
  samples is the **transfer set**. Report both, separately, at every round.
- **Frozen-model control:** the Stage-1 model given the same number of
  attempts, with **no retraining**. "RL discovered N theorems" means nothing
  without the number the frozen model discovers by resampling alone.
- **Found-proof-length histogram** per round: for every theorem solved, the
  length of the proof actually written. Your headline "how far beyond 6" is
  the longest length at which the model wrote **≥ 5 distinct verified proofs**
  (the *robust frontier*), not the single longest proof.
- **In-distribution tracking:** Stage-1 held-out greedy solve rate after every round.
- **Seeds or error bars:** at n = 500 theorems the SE of a difference between
  two solve rates is ≈ 3pp; do not read anything smaller than that as real.
  Two seeds of your main comparison are worth more than a third method.

Your write-up should say what you expected to happen, what did, and what you
think limits the frontier — with evidence, not adjectives. If something you
tried did not work, say so and say why you think so.

A warning about the difficulty label: a theorem you generated with a 12-line
proof may have a 5-line proof. "Solved a 12-line theorem" is not evidence of a
12-line proof. Report what was written.

## 4. Stage 3 — evaluation

Evaluate your final model (and your Stage-1 model, for contrast) on:

1. Your Stage-1 held-out set (length ≤ 6).
2. Your transfer set (7–16 lines), by length.
3. **Validation set** `targets/validation_36.jsonl`: solve rate split by the
   `bin` field (`python prove.py ... --in targets/validation_36.jsonl --out out.jsonl`
   then `python eval_targets.py --proofs out.jsonl`) — `"<=6"` (12 sequents)
   vs `">6"` (24). `min_lines_ub` is an *upper bound* on the shortest proof
   (from a bounded prover, tightened by the shortest proof any model of ours
   has found); if you find a shorter proof than the bound, say so — it is a
   real result. This is the set to look at closely: individual outputs,
   failure modes, proof lengths. **Do not train on it or on renamings of it.**
   Reference points: the best 4-layer model in our own runs (cap 8, several
   RL protocols) solves 15/36; `explosion` (`( P & ( ~ P ) ) ⊢ Q`, 5 lines)
   went unsolved by every 4-layer model.
4. **Test set** `targets/test_short_prompts.jsonl` (≤6-line theorems) and
   `targets/test_long_prompts.jsonl` (longer): run `prove.py` on each and
   `python score_test.py <out.jsonl>`. You get one number per file; report
   both for your Stage-1 model and your final model. **Treat it like a
   leaderboard: no per-theorem inspection, no tuning against it.** The
   verifier is in your hands, so this is a rule rather than a lock — we hold
   a further set you never see, run your `prove.py` on it ourselves, and
   compare. We also audit overlap with your training set.

Show at least: one table (rows = models, columns = the sets above), the
per-length curve for the transfer set with the frozen-model control, and the
found-length histogram across RL rounds. A proof the model wrote for a
textbook theorem it could not prove before RL, pretty-printed, is welcome —
labelled as an existence proof, not a rate.

## 5. What to submit

A single repository (zip or git URL) containing:

- `README.md` — how to reproduce every number (commands, seeds, hardware,
  wall-clock).
- `writeup.md` — see §6. Figures in-line.
- `data/` — training set, held-out set, RL-target and transfer pools (jsonl,
  one record per theorem with at least `{"thm", "text" or "prompt", "n_lines"}`).
- `prove.py` matching `submission_template/prove.py` **exactly** (arguments,
  output schema, no verifier-in-the-loop), plus `requirements.txt` and the
  checkpoint(s) it needs (Stage-1 and final).
- `numbers.md` — every number that appears in the write-up, with the file it
  came from.
- `log.md` — a short dated log of what you tried, in order, including dead
  ends. Honest beats tidy.

## 6. The write-up

`writeup.md` opens with an executive summary of **≤ 600 words plus figures**
that stands on its own: the question, 2–4 findings each supported by one
figure with self-explanatory axes, and the honest headline for "how far past
6". Then a section per stage with enough method detail to follow without the
code (data generation, splits, hyper-parameters, budgets), a limitations
section, and what you would do next with another week.

We read every executive summary in full and the rest as needed. Bullets, tables,
clear figures. Use the positive voice ("a round is …", not "a round is not …").

## 7. What we are looking for

- **Correctness.** Every proof you count is verifier-valid; splits are
  disjoint; the cap is respected; numbers in the text match `numbers.md`.
- **Controls and calibration.** Frozen-model control, transfer set separate
  from RL targets, in-distribution tracked, effect sizes vs. their SE.
- **Insight.** You looked at failures and found *why* the model stops where it
  stops. There is at least one non-obvious barrier in this setting; finding
  and fixing (or clearly characterising) it is worth more than any amount of
  hyper-parameter tuning.
- **Honesty.** Negative results stated as negative, with a mechanism offered.
  Overclaiming a 2pp gain is worse than reporting it as noise.
- **Taste and prioritisation.** One well-tested idea beats five untested ones.
- **Communication.** The executive summary tells the story with figures.

## 8. Rules and FAQ

- You may use AI coding assistants; say so and say for what. The judgement
  calls — what to measure, what to believe — must be yours and must be
  explained in your own words.
- Do not use any external proof data, provers, or LLM-written proofs for
  training. Writing your own search-based prover for *evaluation or analysis*
  is fine if you say so; using it to generate training data is not.
- Q: *Can I change the tokenisation?* Yes, provided the emitted proofs decode
  to the `spec.md` format. Say what you did and why.
- Q: *Can I change the model architecture?* Yes (positional scheme, depth,
  width). Keep it a from-scratch decoder; report parameter count.
- Q: *Is the cap on the theorem or on the proof?* On the proof you train on.
  The theorem may well have a longer or shorter proof than the one you wrote.
- Q: *How is "length" counted when I train?* Number of lines in the training
  proof, as `verify_text` returns it. Every training record must have ≤ 6.
- Q: *What if I run out of time?* Submit what you have with the log. A clean
  Stage 1 and a clear partial Stage 2 with a control beats a rushed Stage 3.

Questions: [contact]. Good luck — this is a small, fully-specified world in
which most of the hard questions about RL on verifiable rewards can be asked
exactly; we hope you enjoy it.






