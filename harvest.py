"""Expert iteration, round 0: sample the frozen model against targets, keep what verifies.

    uv run python harvest.py --ckpt ckpts/frozen/stage1_nope.pt --in data/long/rl_targets.jsonl \
        --k 32 --temperature 1.0 --out data/harvest/nope.jsonl

This is also the **frozen-model control**: the number of distinct theorems a non-retrained model
solves given k attempts each. Any later claim of the form "RL discovered N theorems" is measured
against the number printed here at the same attempt budget.

Only theorems from the RL-target pool belong here. Sampling the transfer pool and training on the
result would destroy the one measurement that answers "did it generalise", so `--in` is asserted
against a deny-list of the evaluation sets.
"""
import argparse
import collections
import json
import os

import torch

from model import load
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens
from ndtok import canonicalize_prompt, decode_safe, rename_tokens, to_ids, to_toks
from prune import prune

FORBIDDEN = ('transfer', 'heldout', 'validation', 'test_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', default='data/long/rl_targets.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=32, help='attempts per theorem')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--max-new', type=int, default=400)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    assert not any(f in a.inp for f in FORBIDDEN), \
        f'{a.inp} is an evaluation set; harvesting from it would contaminate the measurement'
    torch.manual_seed(a.seed)

    model, _ = load(a.ckpt, a.device)
    targets = [json.loads(l) for l in open(a.inp)]
    canon = [canonicalize_prompt(r['prompt']) for r in targets]

    kept, solved_thms, tried = {}, set(), 0
    written = collections.Counter()
    flat = [(i, j) for i in range(len(targets)) for j in range(a.k)]
    for s in range(0, len(flat), a.batch):
        chunk = flat[s:s + a.batch]
        gens = model.generate([to_ids(canon[i][0].split()) for i, _ in chunk],
                              max_new=a.max_new, temperature=a.temperature)
        for (i, _), g in zip(chunk, gens):
            r, inv = targets[i], canon[i][1]
            tried += 1
            proof = decode_safe(r['prompt'], rename_tokens(to_toks(g), inv))
            ok, _, n_lines = verify_text(r['prompt'] + ' ' + proof)
            if not ok:
                continue
            solved_thms.add(r['key'])
            written[n_lines] += 1
            pr = prune(r['prompt'], proof)
            lines = parse_proof_tokens(proof.split())
            # dedupe on the pruned proof: two samples differing only in dead lines are one proof
            kept.setdefault((r['key'], pr['proof']), {
                'thm': r['thm'], 'prompt': r['prompt'], 'proof': proof, 'n_lines': n_lines,
                'pruned_proof': pr['proof'], 'pruned_lines': pr['pruned'], 'dead_lines': pr['dead'],
                'max_depth': max(ln['depth'] for ln in lines), 'key': r['key'],
                'rules': [ln['rule'] for ln in lines if ln['rule'] != 'PR'],
                'gen_lines': r.get('gen_lines'), 'source': os.path.basename(a.ckpt),
                'temperature': a.temperature})
        print(f'  {s + len(chunk)}/{len(flat)} attempts, {len(kept)} distinct proofs, '
              f'{len(solved_thms)} theorems', flush=True)

    recs = sorted(kept.values(), key=lambda r: (r['n_lines'], r['key']))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')

    distinct_w, distinct_p = collections.defaultdict(set), collections.defaultdict(set)
    for r in recs:
        distinct_w[r['n_lines']].add(r['proof'])
        distinct_p[r['pruned_lines']].add(r['pruned_proof'])
    front = lambda d: max((L for L in d if len(d[L]) >= 5), default=None)
    dead = sum(r['dead_lines'] for r in recs)
    print(f'\n{a.ckpt}  k={a.k}  T={a.temperature}  attempts={tried}')
    print(f'FROZEN-MODEL CONTROL: {len(solved_thms)}/{len(targets)} distinct theorems solved')
    print(f'kept {len(recs)} distinct verified proofs (distinct after pruning); by written length '
          f'{dict(sorted(collections.Counter(r["n_lines"] for r in recs).items()))}')
    print(f'  by pruned length {dict(sorted(collections.Counter(r["pruned_lines"] for r in recs).items()))}; '
          f'{dead} dead lines in {sum(r["dead_lines"] > 0 for r in recs)} proofs')
    print(f'robust frontier (>=5 distinct verified): written {front(distinct_w)}, pruned {front(distinct_p)}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
