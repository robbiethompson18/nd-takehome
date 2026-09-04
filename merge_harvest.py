"""Union of several harvest files, deduped by (theorem, proof).

    uv run python merge_harvest.py data/harvest/r*.jsonl --out data/harvest/all.jsonl

A theorem may be proved several different ways; every distinct proof is kept, since they are
distinct training examples.  Prints the cumulative frontier so a round-over-round curve of
"what the model has managed to write so far" falls out of the loop for free.
"""
import argparse
import collections
import json

from prune import prune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    recs = {}
    for path in a.files:
        for line in open(path):
            r = json.loads(line)
            if 'pruned_proof' not in r:          # harvest files from before pruning existed
                pr = prune(r['prompt'], r['proof'])
                r.update(pruned_proof=pr['proof'], pruned_lines=pr['pruned'], dead_lines=pr['dead'])
            recs.setdefault((r['key'], r['pruned_proof']), r)
    merged = sorted(recs.values(), key=lambda r: (r['n_lines'], r['key']))
    with open(a.out, 'w') as f:
        for r in merged:
            f.write(json.dumps(r) + '\n')

    by_len = collections.Counter(r['n_lines'] for r in merged)
    distinct_w, distinct_p = collections.defaultdict(set), collections.defaultdict(set)
    for r in merged:
        distinct_w[r['n_lines']].add(r['proof'])
        distinct_p[r['pruned_lines']].add(r['pruned_proof'])
    front = lambda d: max((L for L in d if len(d[L]) >= 5), default=None)
    print(f'merged {len(a.files)} files -> {len(merged)} distinct proofs (distinct after pruning) over '
          f'{len({r["key"] for r in merged})} theorems')
    print(f'  by written length {dict(sorted(by_len.items()))}')
    print(f'  by pruned length  {dict(sorted(collections.Counter(r["pruned_lines"] for r in merged).items()))}')
    print(f'  cumulative harvest frontier: written {front(distinct_w)}, pruned {front(distinct_p)}')
    print(f'  wrote {a.out}')


if __name__ == '__main__':
    main()
