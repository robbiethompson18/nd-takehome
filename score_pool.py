"""Score a prove.py output against a target pool.

    uv run python score_pool.py --proofs out.jsonl --pool data/long/transfer.jsonl

Reports solve rate with a Wilson interval, the breakdown by `gen_lines` (the generator's proof
length, an upper bound on the shortest proof -- not a difficulty), the histogram of lengths the
model actually WROTE, the same after PRUNING dead lines (prune.py: lines nothing cites, which the
verifier accepts but which are padding), and the robust frontier: the longest length at which the
model produced at least 5 distinct verified proofs.  The frontier is reported on both written and
pruned lengths; the pruned one is the headline for "how far past 6", with distinctness judged on
the pruned proof so two proofs differing only in padding count once.
"""
import argparse
import collections
import json

from eval_targets import wilson
from nd_verify import verify_text
from prune import prune


def score(proofs, pool):
    ref = {r['prompt']: r for r in pool}
    ok, n, written, pruned, reasons = (collections.Counter(), collections.Counter(),
                                       collections.Counter(), collections.Counter(),
                                       collections.Counter())
    distinct_w, distinct_p = collections.defaultdict(set), collections.defaultdict(set)
    dead = collections.Counter()          # dead lines, total lines, proofs with any dead line
    for r in proofs:
        L = ref[r['prompt']].get('gen_lines') or ref[r['prompt']].get('n_lines')
        good, reason, n_lines = verify_text(r['prompt'] + ' ' + r['proof'])
        n[L] += 1
        ok[L] += good
        if good:
            pr = prune(r['prompt'], r['proof'])
            written[n_lines] += 1
            pruned[pr['pruned']] += 1
            distinct_w[n_lines].add(r['proof'])
            distinct_p[pr['pruned']].add(pr['proof'])
            dead['lines'] += pr['dead']
            dead['total'] += n_lines
            dead['proofs'] += pr['dead'] > 0
        else:
            reasons[reason.split('(')[0].strip()] += 1
    return ok, n, written, pruned, distinct_w, distinct_p, dead, reasons


def frontier(distinct):
    return max((L for L in distinct if len(distinct[L]) >= 5), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proofs', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    proofs = [json.loads(l) for l in open(a.proofs)]
    pool = [json.loads(l) for l in open(a.pool)]
    ok, n, written, pruned, distinct_w, distinct_p, dead, reasons = score(proofs, pool)

    k, m = sum(ok.values()), sum(n.values())
    lo, hi = wilson(k, m)
    print(f'{a.label or a.proofs} vs {a.pool}')
    print(f'  solved {k}/{m} = {k/m:.4f}  [{lo:.4f},{hi:.4f}]')
    print('  by gen_lines: ' + ' '.join(f'{L}:{ok[L]}/{n[L]}' for L in sorted(n)))
    print(f'  verified, by length WRITTEN: {dict(sorted(written.items()))}')
    print(f'  verified, by length PRUNED:  {dict(sorted(pruned.items()))}')
    print(f'  dead lines: {dead["lines"]}/{dead["total"]} lines in {dead["proofs"]}/{k} verified proofs')
    print(f'  ROBUST FRONTIER (>=5 distinct verified): written {frontier(distinct_w)}, '
          f'pruned {frontier(distinct_p)}  <- headline')
    print(f'  failures: {dict(reasons.most_common(4))}')


if __name__ == '__main__':
    main()
