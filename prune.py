"""Remove dead lines from a verified proof.

The verifier checks that every line is *legal*, not that it is *used*.  A model can pad a
proof with lines nothing cites and still pass.  `prune` keeps only the dependency cone of
the conclusion (plus the PR block, which must match the prompt), renumbers, and re-verifies.

    from prune import prune
    r = prune(prompt, proof)     # proof = spec body 'N1 ... QED'
    r['written'], r['pruned'], r['dead'], r['proof']

Written length is what the README asks us to report; pruned length is what we use for the
robust frontier and for deciding whether two found proofs are distinct.  A proof the pruner
cannot shorten to a verifier-accepted proof is returned unchanged with dead = 0 and a
'note' (this would indicate a pruner bug).

    uv run python prune.py proofs.jsonl        # records with {"prompt", "proof"}; prints a
                                               # written -> pruned table and the dead-line rate
"""
import collections
import json
import sys

from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens
from ndtok import fmt, parse_prompt

BOX_RULES = {'IMPI', 'NEGI'}


def prune(prompt, proof):
    ptoks, btoks = prompt.split(), proof.split()
    ok, _, written = verify_text(' '.join(ptoks + btoks))
    if not ok:
        return {'proof': proof, 'written': written, 'pruned': written, 'dead': 0, 'note': 'invalid'}
    lines = parse_proof_tokens(btoks)
    by_idx = {ln['idx']: ln for ln in lines}

    # box structure: for each AS line, the last line of its box (what the verifier calls box_lines)
    keep = {ln['idx'] for ln in lines if ln['rule'] == 'PR'}
    todo = [lines[-1]['idx']]
    while todo:
        i = todo.pop()
        if i in keep and i != lines[-1]['idx']:
            continue
        keep.add(i)
        ln = by_idx[i]
        refs = ln['refs']
        if ln['rule'] in BOX_RULES:
            boxes, singles = [tuple(refs)], []
        elif ln['rule'] == 'ORE':
            boxes, singles = [(refs[1], refs[2]), (refs[3], refs[4])], [refs[0]]
        else:
            boxes, singles = [], refs
        todo += singles
        for s, e in boxes:
            keep.add(s)            # the AS line
            todo.append(e)         # the box's conclusion; its cone stays inside the box

    if len(keep) == len(lines):
        return {'proof': proof, 'written': written, 'pruned': written, 'dead': 0}

    renum = {old: new for new, old in enumerate(sorted(keep), start=1)}
    out = []
    for ln in lines:
        if ln['idx'] not in keep:
            continue
        out += [f"N{renum[ln['idx']]}"] + ['|'] * ln['depth'] + fmt(ln['formula']) + [':', ln['rule']]
        out += [f'N{renum[r]}' for r in ln['refs']] + [';']
    pruned = ' '.join(out + ['QED'])
    ok, reason, n = verify_text(' '.join(ptoks) + ' ' + pruned)
    if not ok:
        return {'proof': proof, 'written': written, 'pruned': written, 'dead': 0,
                'note': f'pruned proof rejected: {reason}'}
    return {'proof': pruned, 'written': written, 'pruned': n, 'dead': written - n}


def main():
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    table = collections.Counter()
    dead_lines = total_lines = n_valid = n_changed = 0
    notes = collections.Counter()
    for r in recs:
        res = prune(r['prompt'], r['proof'])
        if res.get('note') == 'invalid':
            continue
        n_valid += 1
        if res.get('note'):
            notes[res['note']] += 1
        table[(res['written'], res['pruned'])] += 1
        dead_lines += res['dead']
        total_lines += res['written']
        n_changed += res['dead'] > 0
    print(f'{n_valid} valid proofs; {n_changed} had dead lines; '
          f'{dead_lines}/{total_lines} lines dead ({100 * dead_lines / max(total_lines, 1):.1f}%)')
    if notes:
        print('notes:', dict(notes))
    print(f'\n{"written":>8} {"pruned":>7} {"n":>7}')
    for (w, p), c in sorted(table.items()):
        print(f'{w:>8} {p:>7} {c:>7}' + ('   <--' if p < w else ''))


if __name__ == '__main__':
    main()
