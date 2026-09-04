"""Describe a proof dataset: what the model will actually see.

    uv run python data_report.py data/train.jsonl [data/heldout.jsonl ...]

Prints, per proof length and overall: rule mix against the 36 reference proofs, box depth,
premises, conclusion shape, prompt size, skeleton (rule-sequence) diversity, unused premises,
and which reference-proof skeletons at <= 6 lines have a structural twin in the data.
"""
import collections
import json
import statistics as st
import sys

from nd_verify.verify import parse_proof_tokens
from ndtok import encode, parse_prompt

RULES = ['AS', 'IMPE', 'IMPI', 'ANDE1', 'ANDE2', 'ANDI', 'ORI1', 'ORI2', 'ORE', 'NEGE', 'NEGI', 'DN', 'BOTE', 'R']


def lines_of(prompt, proof):
    toks = (prompt + ' ' + proof).split()
    _, _, i = parse_prompt(toks)
    return parse_proof_tokens(toks[i:])


def skeleton(lines):
    """Rules in order with box depth, PR dropped: the proof's shape."""
    return tuple((ln['depth'], ln['rule']) for ln in lines if ln['rule'] != 'PR')


def shape(f):
    return 'F' if f == ('bot',) else {'atom': 'atom', 'not': '~', 'and': '&', 'or': 'v', 'imp': '>'}[f[0]]


def unused_premises(lines):
    cited = set()
    for ln in lines:
        cited.update(ln['refs'])
    return sum(1 for ln in lines if ln['rule'] == 'PR' and ln['idx'] not in cited)


def pct_table(title, rows, cols):
    print(f'\n{title}')
    print('  ' + ' ' * 10 + ''.join(f'{c:>7}' for c in cols))
    for name, counter, total in rows:
        print(f'  {name:<10}' + ''.join(f'{100 * counter[c] / max(total, 1):7.1f}' for c in cols))


def main():
    recs = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
    ref = [json.loads(l) for l in open('targets/validation_36_reference_proofs.jsonl')]
    by_len = collections.defaultdict(list)
    for r in recs:
        by_len[r['n_lines']].append(r)

    # --- per-length structure -----------------------------------------------------------
    print(f'{len(recs)} records from {sys.argv[1:]}')
    print(f'\n{"len":>4} {"n":>7} {"prem 0/1/2/3":>14} {"boxes>=1":>9} {"depth2+":>8} {"concl atom/F/~/&/v/>":>22} '
          f'{"prompt tok med/p90":>19} {"skeletons":>10} {"unused PR":>10}')
    all_lines = {}
    for L in sorted(by_len):
        rs = by_len[L]
        prem = collections.Counter(min(r['n_premises'], 3) for r in rs)
        boxes = sum(1 for r in rs if r['boxes'] > 0)
        deep = sum(1 for r in rs if r['max_depth'] >= 2)
        concl = collections.Counter()
        ptok, skel, unused = [], set(), 0
        for r in rs:
            toks = r['prompt'].split()
            premises, c, _ = parse_prompt(toks)
            concl[shape(c)] += 1
            ptok.append(len(toks) - 3)
            ls = lines_of(r['prompt'], r['proof'])
            all_lines[r['key']] = ls
            skel.add(skeleton(ls))
            unused += unused_premises(ls)
        ptok.sort()
        n = len(rs)
        print(f'{L:>4} {n:>7} {"/".join(str(prem[k]) for k in range(4)):>14} {100*boxes/n:>8.0f}% {100*deep/n:>7.1f}% '
              f'{"/".join(str(round(100*concl[k]/n)) for k in ["atom","F","~","&","v",">"]):>22} '
              f'{ptok[n//2]:>9}/{ptok[int(.9*n)]:<9} {len(skel):>10} {unused:>10}')

    # --- rule mix vs reference -----------------------------------------------------------
    def rule_counter(items):
        c = collections.Counter()
        for prompt, proof in items:
            _, body = encode(prompt + ' ' + proof)
            c.update(t for i, t in enumerate(body) if i > 0 and body[i - 1] == ':')
        c['R'] += c.pop('REIT', 0)
        return c
    ours = rule_counter((r['prompt'], r['proof']) for r in recs)
    ref_short = rule_counter((r['prompt'], r['reference_proof']) for r in ref if r['reference_lines'] <= 6)
    ref_all = rule_counter((r['prompt'], r['reference_proof']) for r in ref)
    pct_table('rule mix, % of rule applications',
              [('ours', ours, sum(ours.values())),
               ('ref <=6', ref_short, sum(ref_short.values())),
               ('ref all', ref_all, sum(ref_all.values()))], RULES)

    # --- box depth vs reference ----------------------------------------------------------
    depth_ours = collections.Counter(r['max_depth'] for r in recs)
    depth_ref = collections.Counter()
    for r in ref:
        depth_ref[max([ln['depth'] for ln in lines_of(r['prompt'], r['reference_proof'])])] += 1
    pct_table('max box depth, % of proofs',
              [('ours', depth_ours, len(recs)), ('ref all', depth_ref, len(ref))], [0, 1, 2, 3, 4])

    # --- do reference shapes exist in our data? -----------------------------------------
    skels = collections.Counter(skeleton(ls) for ls in all_lines.values())
    print('\nreference proofs at <= 6 lines: does their exact skeleton (rules + depths) occur in the data?')
    for r in sorted(ref, key=lambda r: r['reference_lines']):
        if r['reference_lines'] > 6:
            continue
        sk = skeleton(lines_of(r['prompt'], r['reference_proof']))
        print(f'  {r["name"]:<28} L={r["reference_lines"]}  {skels.get(sk, 0):>6} twins   '
              + ' '.join(f'{"|"*d}{rule}' for d, rule in sk))

    # --- skeleton concentration ------------------------------------------------------------
    print('\nskeleton concentration: share of proofs covered by the top-k skeletons')
    for L in sorted(by_len):
        c = collections.Counter(skeleton(all_lines[r['key']]) for r in by_len[L])
        tot = sum(c.values())
        top = c.most_common()
        cum = [sum(v for _, v in top[:k]) / tot for k in (1, 5, 20)]
        print(f'  len {L}: {len(c):>5} skeletons; top1 {cum[0]:.0%}  top5 {cum[1]:.0%}  top20 {cum[2]:.0%}   '
              f'most common: {" ".join(f"{chr(124)*d}{rule}" for d, rule in top[0][0])}')


if __name__ == '__main__':
    main()
