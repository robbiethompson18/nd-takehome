"""Stage 1: supervised training of a small decoder on cap-6 proofs.

    uv run python train.py --pos nope --batch 1024 --steps 6000 --out ckpts/stage1_nope.pt

Loss is on the proof body only; the prompt is context.  Held-out evaluation is greedy decoding
scored by the verifier -- a proof counts only if `verify_text` accepts it *for the prompted
sequent* -- broken down by the length of the reference proof, with Wilson intervals.
"""
import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from eval_targets import wilson
from model import Config, Model, load, save
from nd_verify import verify_text
from ndtok import PAD, decode_safe, encode, to_ids, to_toks


def load_split(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        p, b = encode(r['prompt'] + ' ' + r['proof'])
        out.append({'p': to_ids(p), 'b': to_ids(b), 'prompt': r['prompt'], 'n_lines': r['n_lines'],
                    'boxed': r['max_depth'] > 0})
    return out


@torch.no_grad()
def val_loss(model, data, device, batch=512):
    """Teacher-forced loss on the held-out split -- the overfitting check that greedy solve
    rate cannot give you (solve rate is 0/1 per theorem and saturates)."""
    model.eval()
    tot = n = 0.0
    for i in range(0, len(data), batch):
        idxs = range(i, min(i + batch, len(data)))
        x, y = make_batch(data, idxs, device)
        logits = model(x)
        m = y != -100
        tot += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum').item()
        n += m.sum().item()
    model.train()
    return tot / n


def make_batch(data, idxs, device):
    """Right-pad; targets are the body tokens only (-100 elsewhere, ignored by cross_entropy).

    Right-padding means causal masking already hides the pads, so no key-padding mask is needed."""
    rows = [data[i] for i in idxs]
    T = max(len(r['p']) + len(r['b']) for r in rows)
    x = np.full((len(rows), T), PAD, dtype=np.int64)
    y = np.full((len(rows), T), -100, dtype=np.int64)
    for i, r in enumerate(rows):
        np_, nb = len(r['p']), len(r['b'])
        x[i, :np_] = r['p']
        x[i, np_:np_ + nb] = r['b']
        y[i, np_ - 1:np_ + nb - 1] = r['b']
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, data, device, batch=512, max_new=192, temperature=0.0):
    """Greedy solve rate by reference-proof length, plus verifier failure reasons."""
    ok, n, reasons, written = collections.Counter(), collections.Counter(), collections.Counter(), []
    box = collections.Counter()          # (boxed, solved) -> count
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        gens = model.generate([r['p'] for r in chunk], max_new=max_new, temperature=temperature)
        for r, g in zip(chunk, gens):
            proof = decode_safe(r['prompt'], to_toks(g))
            good, reason, n_lines = verify_text(r['prompt'] + ' ' + proof)
            n[r['n_lines']] += 1
            ok[r['n_lines']] += good
            box[(r['boxed'], bool(good))] += 1
            if good:
                written.append(n_lines)
            else:
                reasons[reason.split('(')[0].strip()] += 1
    tot_ok, tot_n = sum(ok.values()), sum(n.values())
    lo, hi = wilson(tot_ok, tot_n)
    return {'solved': tot_ok, 'n': tot_n, 'rate': tot_ok / max(tot_n, 1), 'ci': (lo, hi),
            'by_len': {L: (ok[L], n[L], wilson(ok[L], n[L])) for L in sorted(n)},
            'by_box': {b: (box[(b, True)], box[(b, True)] + box[(b, False)]) for b in (False, True)},
            'reasons': reasons.most_common(6), 'written_lens': collections.Counter(written)}


def fmt_eval(e):
    per = '  '.join(f'{L}:{k}/{m}' for L, (k, m, _) in e['by_len'].items())
    (fk, fn), (bk, bn) = e['by_box'][False], e['by_box'][True]
    box = f'flat {fk}/{fn} boxed {bk}/{bn}'
    return f"solve {e['rate']:.3f} [{e['ci'][0]:.3f},{e['ci'][1]:.3f}] n={e['n']} | {per} | {box}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.jsonl')
    ap.add_argument('--val', default='data/heldout.jsonl')
    ap.add_argument('--out', default='ckpts/stage1.pt')
    ap.add_argument('--pos', default='nope', choices=['nope', 'rope'])
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--n-train', type=int, default=0, help='subsample the training set (0 = all)')
    ap.add_argument('--init', default=None, help='fine-tune from this checkpoint instead of scratch')
    ap.add_argument('--extra', default=None,
                    help='comma-separated jsonl of extra proofs (e.g. an expert-iteration harvest)')
    ap.add_argument('--extra-frac', type=float, default=0.25,
                    help='fraction of every batch drawn from --extra. With a harvest of ~100 '
                         'proofs this is the real knob, not how many times you copy them.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--eval-n', type=int, default=1000, help='held-out theorems per greedy eval')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    train, val = load_split(a.train), load_split(a.val)
    if a.n_train:
        rs = np.random.RandomState(a.seed)     # the split files are sorted by length, so a
        keep = rs.choice(len(train), a.n_train, replace=False)   # prefix would be length-biased
        train = [train[i] for i in keep]
    # same reason: stride the eval subsample so every length bucket is represented
    eval_set = val[::max(1, len(val) // a.eval_n)][:a.eval_n]
    extra = []
    for path in (a.extra.split(',') if a.extra else []):
        extra += load_split(path)
    pool = train + extra                      # indices >= len(train) are the extra pool
    n_extra = int(a.batch * a.extra_frac) if extra else 0

    cfg = Config(n_layer=a.layers, d_model=a.dim, n_head=a.heads, max_len=a.max_len, pos=a.pos)
    model = Model(cfg).to(a.device)
    if a.init:
        init, _ = load(a.init, a.device)
        assert init.cfg.pos == a.pos, f'--init is {init.cfg.pos}, --pos is {a.pos}'
        model.load_state_dict(init.state_dict())
        print(f'initialised from {a.init}')
    if a.compile:
        model = torch.compile(model)
    raw = getattr(model, '_orig_mod', model)
    print(f'{raw.n_params()/1e6:.2f}M params | pos={a.pos} | {len(train)} train / {len(val)} val '
          f'| device={a.device} | epochs={a.steps * a.batch / len(train):.1f}'
          + (f' | extra {len(extra)} at {a.extra_frac:.0%} of batch '
             f'({n_extra}/batch, each seen ~{a.steps * n_extra / max(len(extra),1):.0f}x)'
             if extra else ''))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95),
                            fused=(a.device == 'cuda'))
    amp = torch.autocast(a.device, dtype=torch.bfloat16) if a.device == 'cuda' \
        else torch.autocast('cpu', enabled=False)
    gen = torch.Generator().manual_seed(a.seed)
    history, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * (step / a.warmup if step < a.warmup
                     else 0.1 + 0.45 * (1 + math.cos(math.pi * step / a.steps)))
        for gparam in opt.param_groups:
            gparam['lr'] = lr
        idxs = torch.randint(len(train), (a.batch - n_extra,), generator=gen).tolist()
        if n_extra:
            idxs += (len(train) + torch.randint(len(extra), (n_extra,), generator=gen)).tolist()
        x, y = make_batch(pool, idxs, a.device)
        with amp:
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0:
            vl = val_loss(model, eval_set, a.device)
            ep = step * a.batch / len(train)
            print(f'step {step:5d}  ep {ep:5.1f}  train {loss.item():.4f}  val {vl:.4f}  '
                  f'lr {lr:.2e}  {(time.time()-t0)/step*1000:.0f} ms/step', flush=True)
            history.append({'step': step, 'epoch': ep, 'train_loss': loss.item(), 'val_loss': vl})
        if step % a.eval_every == 0 or step == a.steps:
            e = evaluate(raw, eval_set, a.device)
            model.train()
            print(f'  [eval @ {step}] {fmt_eval(e)}  reasons {e["reasons"]}', flush=True)
            history.append({'step': step, **{k: v for k, v in e.items()
                            if k in ('solved', 'n', 'rate', 'ci')}})

    final = evaluate(raw, val, a.device)
    print(f'FINAL (full held-out) {fmt_eval(final)}')
    for L, (k, m, (lo, hi)) in final['by_len'].items():
        print(f'  len {L}: {k}/{m} = {k/m:.3f}  [{lo:.3f},{hi:.3f}]')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    save(a.out, raw, {'args': vars(a), 'history': history,
                      'final': {k: v for k, v in final.items() if k != 'written_lens'}})
    print(f'wrote {a.out} in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
