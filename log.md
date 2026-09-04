# log.md

What we tried, in order, dead ends included. All of it on 2026-09-03; times are approximate and
taken from file timestamps and run logs. Two agents worked the repo in parallel for much of the
afternoon, so some of these overlap rather than strictly follow one another.

## Format and generator

**~01:00 — Tokenisation.** Decided to drop line-index tokens entirely. A model trained on ≤6-line
proofs has never emitted `N7`, so with `N<i>` in the alphabet it cannot cite line 1 from line 12 no
matter how good it is. Citations became content-addressed: the model writes the *formula* it is
citing where the citation is not already determined by the rule, and the decoder resolves it to the
most recent matching line in scope. This is the single largest design decision and it has a cost we
only measured much later (see "bad line cite" below).

**~02:00 — Backward generator.** Forward random walks were rejected on inspection before being
built: they produce dead lines, and they essentially never produce a derivation of `F` (you would
need `A` and `( ~ A )` to appear in scope by chance) or a box whose hypothesis is actually used. So
we went goal-directed instead. Harder to write; Claude wanted to anyway.

**~19:30 — `R` is a noun and a verb.** `ndtok.VOCAB` contained `'R'` twice, once as the atom and
once as the reiteration rule; the dict comprehension silently collapsed them, so index 8 was a dead
logit and atom-`R` and rule-`R` shared an embedding. Renamed the rule to `REIT` model-facing and
added an assert. Harmless in practice — both decoded to the same string — but free to fix.

**~20:00 — Theorem identity.** First cut deduped on the `thm` string. That misses premise
reorderings, so `P , Q ⊢ P` and `Q , P ⊢ P` were separate theorems and could straddle the split.
Replaced with `gen.theorem_key`, which folds atom renaming *and* premise order. It caught 669
duplicate theorems, 9 of which were reorderings of a held-out theorem sitting in train — a real
leak, found only because the key was tightened.

## Stage 1

**~20:00 — Data.** 900 s of generation gives 142k train / 2k held-out. Lengths 2 and 3 saturate:
the generator runs out of distinct 2-line theorems at about 7.5k no matter how long it runs, so the
set is not flat over length. We took that as evidence that data *quantity* was not going to be the
binding constraint.

**~20:40 — Architecture and the NoPE bet.** 4 layers, d=256, 3.17M params, matching the reference.
The one non-default choice was no positional encoding at all: nothing in the weights is then indexed
by absolute position, so a model trained at ≤141 tokens has nothing that breaks at 300. Trained NoPE
and RoPE side by side on two H100s.

**Dead end: the positional scheme bought nothing.** 99.55% vs 99.75% held-out, robust frontier 7 for
both, no separation anywhere we looked. It was the main architectural bet of Stage 1 and it did not
pay. The honest reading is that the tokenisation change had already removed the length-extrapolation
blocker that NoPE was meant to address, and what remains — the model's prior over when to stop — is
not a positional problem.

**Infrastructure friction, ~30 min and ~$0.26 of it.** Three failed pod launches in a row: RunPod's
edge 403s python-urllib's default User-Agent; the pytorch image has no `rsync`, so file transfer had
to become a tar stream over ssh; and `json.dumps` escaped newlines to a literal `\n`, which bash read
as two characters (`bash: line 1: nset: command not found`). Each failure terminated its own pod in a
`finally`, so the cost stayed in cents. Worth writing down because none of it was visible in advance.

**Measurement bug: the held-out subsample.** Mid-training evaluation read `val[:1024]` of a
length-ordered file, so it covered lengths 2-4 only and read high, and made held-out loss sit
*below* train loss. Fixed by striding, but after these two runs had finished — the loss and
solve-rate-over-epochs charts still carry the caveat rather than being re-run.

## Past the cap

**~21:00 — The evaluation pools.** There was no set that could measure how far past 6 the model
gets: our held-out is capped at 6 by construction, and the only >6 theorems available were the 24 in
`validation_36`, where 0/24 has a 95% interval of [0, 0.14] and is nearly uninformative. Built our
own 7-16 line pools instead, 930 RL targets and 931 transfer, disjoint by key from everything.

**Dead end, and the most useful finding of the day: the pool's difficulty labels are inflated.** The
frozen model's solve rate was non-zero out at generating length 14, which looked like occasional
long-proof capability. It is not. Every solved target labelled 11 lines or more was written in 5-7
lines. The generator's `budget` knob makes long *derivations*, not hard *theorems*: a premise like
`( ( P & Q ) > S ) & ( P & Q )` carries an implication and its own antecedent, so two ANDE and one
IMPE finish it in six lines whatever route the generator took, and a premise that contradicts
something two steps away lets ex falso reach any conclusion at all. Of the solves at generating
length 7, 80% really did take 7 lines; at 8 it is 15%; at 9 and beyond, 0%.

Consequences we accepted rather than fixed: solved theorems are now filed under
`min(generating length, length written)`, which tightens the numerator but not the denominator —
an unsolved target has no known shortest proof, so it keeps its inflated label. The real fix is a
bounded prover giving shortest-proof lengths for the whole pool, which we did not write. Headline
claims therefore use the robust frontier (a count of verified proofs by length written, with no
denominator to corrupt) rather than any per-length rate.

**Correction we had to make.** On the 36-theorem validation set the model appeared never to write
more than 6 lines, which suggested a hard stop at the cap. At n=1,861 that was wrong: it writes 7
lines on about a third of long targets, and occasionally 8-10. It goes past the cap freely and is
usually wrong when it does. Small-n conclusions did not survive.

**Failure mode.** 2,818 of 3,554 rejections are `bad line cite` — the model names a witness formula
that is not in scope, the decoder cannot resolve it, and it fills the citation with the line's own
index, which the verifier refuses. That is a failure of our tokenisation choice as much as of the
model: it hallucinates a premise it does not have, and the format converts that into an unresolvable
reference rather than a wrong-but-legible line.

**Dead end: rebalancing the generator towards boxes.** The three rules that open a subproof are the
rare ones (IMPI 7.8% of proofs, ORE 0.6%, NEGI 0.2%) and long proofs are 94% boxed, so scarce
subproof practice looked like the obvious bottleneck. Turning up `p_intro`, `p_ore` and `p_vacuous`
moved boxed share the *wrong* way, 29% → 24%. Hypothesis: boxed theorems occupy a smaller distinct
space, so they saturate first and a large bucket fills with flat proofs from the tail. A separate
depth-two generation pipeline did produce ~2k nested examples, and putting them in the mix barely
moved the needle on writing 7-line proofs.

## Stage 2

**~22:00 — Expert iteration**, three rounds, NoPE only (RoPE was killed after Stage 1 once it was
clear the two were indistinguishable). Harvest what the model proves on the RL targets, mix it back
in, retrain, repeat.

Result: transfer 4.9% → 10.4%, a 5.5pp gain against roughly 1.3pp of standard error, on a pool the
loop never sampled — so it is not memorisation of the harvested proofs. The frozen model given the
same attempt budget instead of retraining goes 75 → 79 → 82 theorems (+7); retraining goes
65 → 115 → 134 (+69).

**Time lost to a logging bug.** `harvest.py` prints `FROZEN-MODEL CONTROL:` for every harvest,
including the loop's own, so `logs/pod_loop.log` reads as though the control went
65 → 75 → 115 → 66 → 134 → 64. It did not: the loop model went 65 → 115 → 134 and the control ran
75 / 66 / 64 per round (cumulative union 75 → 79 → 82). Recorded in `CODE_SMELL.md`; the numbers in
`numbers.md` are sourced from the harvest files rather than from that log.

**The frontier moved by one line.** 7 before, 8 after, on both pools. L − P = 1. The model has never
written a nine-line proof, in any round, on either pool, which is the number we would most like to
understand and do not.

## What we would do next

1. **A bounded prover** for shortest-proof labels over the 7-16 pools. Without it, no per-length
   rate past 8 means what it appears to mean, and this contaminates any future L − P claim too.
2. **Attack `bad line cite` directly**, since it is three quarters of all failures — either by
   letting the decoder fail softly instead of emitting a self-reference, or by giving the model a
   way to check scope before it commits to a witness.
3. **Understand the nine-line wall.** Three rounds of expert iteration moved 7→8 and then stopped
   dead. Whether that is a data, search or capacity limit is untested.
