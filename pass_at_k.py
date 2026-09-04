"""pass@k on a target pool, broken down by generating length.

    uv run python pass_at_k.py --ckpt ckpts/frozen/stage1_nope.pt --in data/long/transfer.jsonl \
        --k 32 --out logs/passk_stage1_transfer.json

A theorem counts as solved if ANY of its k samples verifies.  Unlike harvest.py this writes no
training data -- only a JSON summary -- so it is safe to run against the transfer pool, which must
never be trained on.  It also records the lengths actually written (pruned), so the frontier can
be computed at the same attempt budget as the solve rate.
"""
import argparse
import collections
import json
import os

import torch

from eval_targets import wilson
from model import load
from nd_verify import verify_text
from ndtok import canonicalize_prompt, decode_safe, rename_tokens, to_ids, to_toks
from prune import prune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--max-new', type=int, default=400)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    model, _ = load(a.ckpt, a.device)
    targets = [json.loads(l) for l in open(a.inp)]
    canon = [canonicalize_prompt(r['prompt']) for r in targets]

    solved = collections.defaultdict(bool)         # target index -> solved by any sample
    written, distinct = collections.Counter(), collections.defaultdict(set)
    flat = [(i, j) for i in range(len(targets)) for j in range(a.k)]
    for s in range(0, len(flat), a.batch):
        chunk = flat[s:s + a.batch]
        gens = model.generate([to_ids(canon[i][0].split()) for i, _ in chunk],
                              max_new=a.max_new, temperature=a.temperature)
        for (i, _), g in zip(chunk, gens):
            r, inv = targets[i], canon[i][1]
            proof = decode_safe(r['prompt'], rename_tokens(to_toks(g), inv))
            ok, _, _ = verify_text(r['prompt'] + ' ' + proof)
            if ok:
                solved[i] = True
                pr = prune(r['prompt'], proof)
                written[pr['pruned']] += 1
                distinct[pr['pruned']].add(pr['proof'])
        print(f'  {s + len(chunk)}/{len(flat)} attempts, {sum(solved.values())} theorems',
              flush=True)

    by_len = collections.defaultdict(lambda: [0, 0])
    for i, r in enumerate(targets):
        L = r.get('gen_lines') or r.get('n_lines')
        by_len[L][0] += bool(solved.get(i))
        by_len[L][1] += 1
    out = {'ckpt': a.ckpt, 'pool': a.inp, 'k': a.k, 'temperature': a.temperature,
           'n': len(targets), 'solved': sum(solved.values()),
           'by_len': {str(L): {'solved': k, 'n': m, 'ci': wilson(k, m)}
                      for L, (k, m) in sorted(by_len.items())},
           'written_pruned': dict(sorted(written.items())),
           'distinct_pruned': {str(L): len(v) for L, v in sorted(distinct.items())},
           'frontier': max((L for L in distinct if len(distinct[L]) >= 5), default=None)}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    lo, hi = wilson(out['solved'], out['n'])
    print(f"\n{a.ckpt} vs {a.inp}  pass@{a.k}: {out['solved']}/{out['n']} = "
          f"{out['solved']/out['n']:.4f} [{lo:.4f},{hi:.4f}]  frontier {out['frontier']}")
    for L, d in out['by_len'].items():
        print(f"  len {L:>2}: {d['solved']:3d}/{d['n']:3d} = {d['solved']/d['n']:.3f}")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
