"""Evaluation pools of theorems whose generating proofs are 7-16 lines.

    uv run python make_eval_pool.py --per-len 200 --seconds 300

These are targets, never training data.  This script has no code path that writes
`train.jsonl` or `heldout.jsonl`; the `hi <= 6` assert in make_data.py stays intact and
un-bypassed, so the cap on what we train on is still guarded by something mechanical.

Writes two disjoint pools, as the exam requires:

  rl_targets.jsonl : the theorems Stage-2 RL is allowed to sample against
  transfer.jsonl   : a pool RL never sees, so "did it generalise" has an answer

Both are keyed disjointly from each other, from data/train.jsonl, from data/heldout.jsonl and
from everything under targets/.

A warning the exam makes explicitly and that these files inherit: `gen_lines` is the length of
the proof the generator happened to build, which is an UPPER BOUND on the shortest proof.  A
theorem here labelled 12 may well have a 5-line proof.  Solving one is not evidence of a 12-line
proof -- only the length the model actually writes is.
"""
import argparse
import collections
import json
import random

import gen
from make_data import read, target_theorems, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lo', type=int, default=7)
    ap.add_argument('--hi', type=int, default=16)
    ap.add_argument('--per-len', type=int, default=200, help='theorems per generating length')
    ap.add_argument('--seed', type=int, default=1, help='not 0: keep it off the training stream')
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--data', default='data', help='split whose keys must be excluded')
    ap.add_argument('--out', default='data/long')
    a = ap.parse_args()
    assert a.lo > 6, 'this script is for the >6 pools; use make_data.py for training data'

    train, held = read(f'{a.data}/train.jsonl'), read(f'{a.data}/heldout.jsonl')
    banned = target_theorems() | {r['key'] for r in train + held}
    print(f'excluding {len(banned)} keys ({len(train)} train + {len(held)} held-out + targets/)')

    got, st = gen.sample(a.per_len, a.lo, a.hi, seed=a.seed, banned=banned, seconds=a.seconds)
    rng = random.Random(a.seed)
    rl, transfer = [], []
    for L in sorted(got):
        recs = got[L]
        rng.shuffle(recs)
        half = len(recs) // 2                     # split each length bucket, so the two pools
        rl += recs[:half]                         # have the same length profile
        transfer += recs[half:]
    for r in rl + transfer:
        r['gen_lines'] = r.pop('n_lines')         # rename: it is an upper bound, not a difficulty

    ks = [{r['key'] for r in p} for p in (rl, transfer, train, held)]
    assert not (ks[0] & ks[1]), 'rl targets and transfer set share a theorem'
    assert not ((ks[0] | ks[1]) & (ks[2] | ks[3])), 'eval pool overlaps the Stage-1 split'

    print(f'{len(rl) + len(transfer)} theorems in {st["seconds"]}s ({st["tries"]} tries)')
    for name, pool in [('rl_targets', rl), ('transfer', transfer)]:
        by_len = collections.Counter(r['gen_lines'] for r in pool)
        boxed = sum(1 for r in pool if r['max_depth'] > 0)
        deep = sum(1 for r in pool if r['max_depth'] >= 2)
        path = f'{a.out}/{name}.jsonl'
        with open(path, 'w') as f:                # write() sorts on n_lines, which we renamed
            for r in sorted(pool, key=lambda x: (x['gen_lines'], x['key'])):
                f.write(json.dumps(r) + '\n')
        print(f'  {path}: {len(pool)} theorems | boxed {boxed/max(len(pool),1):.1%} '
              f'(depth>=2 {deep/max(len(pool),1):.1%}) | by gen_lines {dict(sorted(by_len.items()))}')
    json.dump({'seed': a.seed, 'lo': a.lo, 'hi': a.hi, 'per_len': a.per_len,
               'n_rl_targets': len(rl), 'n_transfer': len(transfer),
               'excluded_keys': len(banned), 'seconds': st['seconds'],
               'note': 'gen_lines is an upper bound on the shortest proof, not a difficulty'},
              open(f'{a.out}/meta.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
