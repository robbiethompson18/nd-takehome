"""Stage-1 data: cap-6 proofs, split train / held-out disjointly by theorem.

    uv run python make_data.py --per-len 40000 --val-per-len 400

Two properties the exam asks about, both by construction:

  * records are deduped on `key` (gen.theorem_key: the theorem up to atom renaming AND premise
    order), so the held-out set contains no renaming or premise-reordering of a training theorem.
  * every sequent appearing in `targets/` (validation_36 and both test files, keyed the same
    way) is dropped from both splits, so neither split overlaps the evaluation sets.

Output: data/train.jsonl and data/heldout.jsonl ("held-out" is the README's word for our own
<= 6 split; "validation" is reserved for the provided targets/validation_36).

Sampling saturates at the short lengths: there are only so many distinct 2-line theorems the
generator can build, so a bucket may come up short of its quota.  The report says which.
"""
import argparse
import collections
import glob
import json
import os
import random

import gen
from ndtok import encode, parse_prompt


KNOBS = {'p_intro': 'decompose the goal (IMPI/NEGI open boxes) instead of eliminating',
         'p_ore': 'use OR-elimination, which opens two boxes',
         'p_bote': 'use ex falso', 'p_vacuous': 'allow IMPI boxes that ignore their hypothesis',
         'p_hyp_minor': 'prefer an enclosing box hypothesis when a subgoal is needed',
         'p_trivial': 'fraction of theorems whose conclusion is already a premise',
         'p_anchor': 'start eliminations from an in-scope formula rather than a fresh premise'}


def read(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def write(path, recs):
    """Sorted by (length, key): stable across appends, diffable, and cheap to inspect.
    Training draws random indices, so file order does not affect batching."""
    recs = sorted(recs, key=lambda r: (r['n_lines'], r['key']))
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    return recs


def key(prompt):
    premises, concl, _ = parse_prompt(prompt.split())
    return gen.theorem_key(premises, concl)


def target_theorems():
    """Keys of every sequent under targets/, to exclude from training."""
    out = set()
    for path in glob.glob('targets/*.jsonl'):
        for line in open(path):
            r = json.loads(line)
            if 'prompt' in r:
                out.add(key(r['prompt']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-len', type=int, default=40000, help='train+val theorems per length')
    ap.add_argument('--val-per-len', type=int, default=400)
    ap.add_argument('--lo', type=int, default=2)
    ap.add_argument('--hi', type=int, default=6, help='the exam cap; do not raise it')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=600, help='sampling budget')
    ap.add_argument('--out', default='data')
    ap.add_argument('--nested', action='store_true',
                    help='sample only box-depth >= 2 proofs (gen.sample nested mode); use with --append')
    ap.add_argument('--append', action='store_true',
                    help='keep the existing split, add only theorems whose key is new. Existing '
                         'held-out records are never touched; --heldout-frac of the new records '
                         'join held-out so the new data class can be measured in-distribution.')
    ap.add_argument('--heldout-frac', type=float, default=0.05,
                    help='append mode: fraction of newly added theorems that go to held-out')
    for k, v in KNOBS.items():                      # generator mix, e.g. --p-ore 0.4
        ap.add_argument(f'--{k.replace("_", "-")}', type=float, default=None, help=f'gen.Gen {k}')
    a = ap.parse_args()
    assert a.hi <= 6, 'training data is capped at 6 lines'
    knobs = {k: getattr(a, k) for k in KNOBS if getattr(a, k) is not None}

    old_train = read(f'{a.out}/train.jsonl') if a.append else []
    old_held = read(f'{a.out}/heldout.jsonl') if a.append else []
    banned = target_theorems() | {r['key'] for r in old_train + old_held}
    if a.append:
        print(f'append mode: {len(old_train)} train + {len(old_held)} held-out already on disk, '
              f'their keys are excluded; knobs {knobs or "(defaults)"}')
    got, st = gen.sample(a.per_len, a.lo, a.hi, seed=a.seed, banned=banned, seconds=a.seconds,
                         knobs=knobs, nested=a.nested)
    rng = random.Random(a.seed)
    train, val = [], []
    print(f'{sum(len(v) for v in got.values())} theorems in {st["seconds"]}s ({st["tries"]} tries), '
          f'{st["banned"]} dropped as already in targets/ or the existing split '
          f'({len(banned)} keys banned)')
    for L in sorted(got):
        recs = got[L]
        rng.shuffle(recs)
        n_val = int(a.heldout_frac * len(recs)) if a.append else min(a.val_per_len, len(recs) // 4)
        val += recs[:n_val]
        train += recs[n_val:]
        toks = [len(encode(r['prompt'] + ' ' + r['proof'])[1]) for r in recs[:200]]
        triv = sum(1 for r in recs if r['trivial']) / len(recs)
        depth = collections.Counter(r['max_depth'] for r in recs)
        short = '' if len(recs) >= a.per_len else f'  SHORT of {a.per_len}'
        print(f'  len {L}: {len(recs)-n_val:6d} train {n_val:5d} val | trivial {triv:5.1%} | '
              f'depth {dict(sorted(depth.items()))} | body tokens med '
              f'{sorted(toks)[len(toks)//2]}{short}')

    prev = json.load(open(f'{a.out}/meta.json')) if a.append and os.path.exists(f'{a.out}/meta.json') \
        else {'rounds': []}
    prev.setdefault('rounds', [])
    for r in train + val:
        r['round'] = len(prev['rounds']) + 1          # 1 = the original split, 2+ = appended
    added, train, val = len(train), old_train + train, old_held + val
    assert not ({r['key'] for r in train} & {r['key'] for r in val}), 'splits share a theorem'
    assert len({r['key'] for r in train}) == len(train), 'duplicate key in train'
    for name, recs in [('train', train), ('heldout', val)]:
        write(f'{a.out}/{name}.jsonl', recs)
        print(f'wrote {a.out}/{name}.jsonl  ({len(recs)} records)')
    boxed = collections.Counter()
    for r in train:
        boxed[(r['n_lines'], r['max_depth'] > 0)] += 1
    print(f'added {added} train records; boxed fraction by length now ' +
          ' '.join(f'{L}:{boxed[(L,True)]/(boxed[(L,True)]+boxed[(L,False)]):.1%}'
                   for L in range(a.lo, a.hi + 1)))
    prev['rounds'].append({'seed': a.seed, 'per_len': a.per_len, 'seconds': st['seconds'],
                           'knobs': knobs, 'nested': a.nested, 'added_train': added,
                           'added_heldout': len(val) - len(old_held)})
    prev.update({'lo': a.lo, 'hi': a.hi, 'n_train': len(train), 'n_heldout': len(val),
                 'target_overlap_dropped': st['banned']})
    json.dump(prev, open(f'{a.out}/meta.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
