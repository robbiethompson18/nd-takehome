#!/usr/bin/env python3
"""Sampling interface (see submission_template/prove.py).

    python prove.py --ckpt ckpts/stage1_nope.pt --in targets/validation_36.jsonl --out out.jsonl

Prompts arrive with arbitrary atom names, but the model only ever saw theorems whose atoms are
in first-occurrence order (P, Q, R, S).  So each prompt is canonicalised before it reaches the
model and the model's atoms are renamed back before decoding.  No verifier in the loop: whatever
the model writes is what gets written out.
"""
import argparse
import json

import torch

from model import load
from ndtok import canonicalize_prompt, decode_safe, rename_tokens, to_ids, to_toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-new', type=int, default=400)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    temp = 0.0 if a.greedy or a.temperature is None else a.temperature
    torch.manual_seed(a.seed)

    model, _ = load(a.ckpt, a.device)
    recs = [json.loads(l) for l in open(a.inp)]
    canon = [canonicalize_prompt(r['prompt']) for r in recs]

    out = []
    for i in range(0, len(recs), a.batch):
        chunk, cchunk = recs[i:i + a.batch], canon[i:i + a.batch]
        gens = model.generate([to_ids(c.split()) for c, _ in cchunk],
                              max_new=a.max_new, temperature=temp)
        for j, (r, (_, inv), g) in enumerate(zip(chunk, cchunk, gens)):
            body = rename_tokens(to_toks(g), inv)
            # targets/ files carry "name"; our own splits key on "key" instead
            name = r.get('name') or r.get('key') or str(i + j)
            out.append({'name': name, 'prompt': r['prompt'],
                        'proof': decode_safe(r['prompt'], body)})

    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {a.out} ({len(out)} proofs, temperature={temp}, ckpt={a.ckpt})')


if __name__ == '__main__':
    main()
