"""Model-facing proof format <-> spec.md format.

The model never sees line indices. A proof body in the model-facing ("compact") form is

    line := '|'*depth <formula> : <RULE> [<witness formula>] ;     ...    QED

and differs from spec.md in three ways:

  * no `N<i>` line indices: lines are numbered consecutively at decode time.
  * no `PR` lines: they are reconstructed from the prompt.
  * citations are by content, not by index.  Most citations are fully determined by the
    rule name and the derived formula (IMPI deriving ( A > B ) can only cite the box that
    assumes A and ends in B).  The only free choice is one *witness* formula:

        ANDE1 / ANDE2 -> the conjunction      IMPE -> the implication
        NEGE          -> the negation         ORE  -> the disjunction

    Every other rule cites nothing.  At decode time each citation resolves to the most
    recent in-scope line (or most recent citable box) whose formula matches.  An
    unresolvable citation is filled with the line's own index, which the verifier rejects.

Why: with `N<i>` in the alphabet, a model trained on <= 6-line proofs has never emitted
`N7` and cannot cite line 1 from line 12.  Copying a formula is position-invariant.
"""
from nd_verify.verify import BOT, RULE_NAMES, ParseError, parse_formula, parse_proof_tokens

# rule -> index (into the spec refs list) of the ref whose formula is the witness
WITNESS = {'ANDE1': 0, 'ANDE2': 0, 'IMPE': 0, 'NEGE': 1, 'ORE': 0}
ARITY = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2, 'ORI1': 1, 'ORI2': 1,
         'ORE': 5, 'NEGE': 2, 'NEGI': 2, 'BOTE': 1, 'DN': 1, 'R': 1}
# spec.md spells both the atom R and the reiteration rule R as `R`.  Model-facing the rule is
# `REIT`, so the two never share an embedding; decode maps it back.
RULE_TOK = {'R': 'REIT'}
TOK_RULE = {v: k for k, v in RULE_TOK.items()}
MODEL_RULES = sorted(RULE_TOK.get(r, r) for r in RULE_NAMES - {'PR'})

VOCAB = (['<pad>', 'THM', ',', 'SEQ', 'PRF', 'QED',
          'P', 'Q', 'R', 'S', 'F', '(', ')', '~', '&', 'v', '>', '|', ':', ';']
         + MODEL_RULES)
assert len(set(VOCAB)) == len(VOCAB), 'duplicate token in VOCAB'
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, QED = STOI['<pad>'], STOI['QED']


def fmt(f):
    """Formula tuple -> spec.md tokens."""
    if f == BOT:
        return ['F']
    if f[0] == 'atom':
        return [f[1]]
    if f[0] == 'not':
        return ['(', '~', *fmt(f[1]), ')']
    op = {'and': '&', 'or': 'v', 'imp': '>'}[f[0]]
    return ['(', *fmt(f[1]), op, *fmt(f[2]), ')']


def parse_prompt(toks):
    """'THM p1 , p2 SEQ c PRF ...' -> (premises, conclusion, index after PRF)."""
    if not toks or toks[0] != 'THM':
        raise ParseError('missing THM')
    i, premises = 1, []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            premises.append(f)
            if toks[i] != ',':
                break
            i += 1
    if toks[i] != 'SEQ':
        raise ParseError('missing SEQ')
    concl, i = parse_formula(toks, i + 1)
    if i >= len(toks) or toks[i] != 'PRF':
        raise ParseError('missing PRF')
    return premises, concl, i + 1


def encode(text):
    """Full spec.md example 'THM ... PRF ... QED' -> (prompt tokens, compact body tokens)."""
    toks = text.split() if isinstance(text, str) else list(text)
    _, _, i = parse_prompt(toks)
    lines = parse_proof_tokens(toks[i:])
    fml = {ln['idx']: ln['formula'] for ln in lines}
    body = []
    for ln in lines:
        if ln['rule'] == 'PR':
            continue
        body += ['|'] * ln['depth'] + fmt(ln['formula']) + [':', RULE_TOK.get(ln['rule'], ln['rule'])]
        w = WITNESS.get(ln['rule'])
        if w is not None:
            body += fmt(fml[ln['refs'][w]])
        body.append(';')
    body.append('QED')
    return toks[:i], body


def decode(prompt, body):
    """(prompt tokens/text, compact body tokens/text) -> spec.md proof body text 'N1 ... QED'.

    Raises ParseError on malformed input; use decode_safe from sampling code."""
    ptoks = prompt.split() if isinstance(prompt, str) else list(prompt)
    btoks = body.split() if isinstance(body, str) else list(body)
    premises, _, _ = parse_prompt(ptoks)

    lines = []                     # dicts: idx depth formula rule refs
    fml, ctx = {}, {}              # idx -> formula, idx -> open-box tuple after the line
    stack, nbox = [], 0
    box_start, box_last, box_depth = {}, {}, {}
    for p in premises:
        idx = len(lines) + 1
        lines.append({'idx': idx, 'depth': 0, 'formula': p, 'rule': 'PR', 'refs': []})
        fml[idx], ctx[idx] = p, ()

    def find_line(idx, pred):
        ci = ctx[idx]
        for j in range(idx - 1, 0, -1):
            cj = ctx[j]
            if cj == ci[:len(cj)] and pred(fml[j]):
                return j
        return None

    def find_box(idx, hyp, end):
        ci = ctx[idx]
        for b in range(nbox, 0, -1):
            if b in ci:
                continue
            s, e = box_start[b], box_last[b]
            parent = ctx[s][:-1]
            if parent != ci[:len(parent)] or lines[e - 1]['depth'] != box_depth[b]:
                continue
            if fml[s] == hyp and fml[e] == end:
                return [s, e]
        return None

    def resolve(idx, rule, G, W):
        eq = lambda x: (lambda f: f == x)
        shape = lambda f, k: isinstance(f, tuple) and f[0] == k
        r = None
        if rule == 'R':
            r = [find_line(idx, eq(G))]
        elif rule == 'ANDI' and shape(G, 'and'):
            r = [find_line(idx, eq(G[1])), find_line(idx, eq(G[2]))]
        elif rule in ('ANDE1', 'ANDE2') and W is not None:
            r = [find_line(idx, eq(W))]
        elif rule == 'IMPE' and shape(W, 'imp'):
            r = [find_line(idx, eq(W)), find_line(idx, eq(W[1]))]
        elif rule == 'IMPI' and shape(G, 'imp'):
            r = find_box(idx, G[1], G[2])
        elif rule == 'ORI1' and shape(G, 'or'):
            r = [find_line(idx, eq(G[1]))]
        elif rule == 'ORI2' and shape(G, 'or'):
            r = [find_line(idx, eq(G[2]))]
        elif rule == 'ORE' and shape(W, 'or'):
            b1, b2 = find_box(idx, W[1], G), find_box(idx, W[2], G)
            r = [find_line(idx, eq(W))] + (b1 or [None, None]) + (b2 or [None, None])
        elif rule == 'NEGE' and shape(W, 'not'):
            r = [find_line(idx, eq(W[1])), find_line(idx, eq(W))]
        elif rule == 'NEGI' and shape(G, 'not'):
            r = find_box(idx, G[1], BOT)
        elif rule == 'BOTE':
            r = [find_line(idx, eq(BOT))]
        elif rule == 'DN':
            r = [find_line(idx, eq(('not', ('not', G))))]
        if r is None:
            r = [None] * ARITY[rule]
        return [idx if x is None else x for x in r]

    i = 0
    while True:
        if i >= len(btoks):
            raise ParseError('missing QED')
        if btoks[i] == 'QED':
            break
        depth = 0
        while i < len(btoks) and btoks[i] == '|':
            depth += 1
            i += 1
        G, i = parse_formula(btoks, i)
        if i >= len(btoks) or btoks[i] != ':':
            raise ParseError('missing :')
        i += 1
        if i >= len(btoks) or btoks[i] not in MODEL_RULES:
            raise ParseError('bad rule name')
        rule = TOK_RULE.get(btoks[i], btoks[i])
        i += 1
        W = None
        if i < len(btoks) and btoks[i] != ';':
            W, i = parse_formula(btoks, i)
        if i >= len(btoks) or btoks[i] != ';':
            raise ParseError('missing ;')
        i += 1

        idx = len(lines) + 1
        # box bookkeeping, mirrors nd_verify.verify
        if rule == 'AS':
            if depth < 1 or depth > len(stack) + 1:
                raise ParseError(f'bad AS depth at line {idx}')
            del stack[depth - 1:]
            nbox += 1
            stack.append(nbox)
            box_start[nbox], box_depth[nbox] = idx, depth
        else:
            if depth > len(stack):
                raise ParseError(f'depth jump at line {idx}')
            del stack[depth:]
        for b in stack:
            box_last[b] = idx
        fml[idx], ctx[idx] = G, tuple(stack)
        lines.append({'idx': idx, 'depth': depth, 'formula': G, 'rule': rule, 'refs': []})
        if rule != 'AS':
            lines[-1]['refs'] = resolve(idx, rule, G, W)

    out = []
    for ln in lines:
        out += [f"N{ln['idx']}"] + ['|'] * ln['depth'] + fmt(ln['formula']) + [':', ln['rule']]
        out += [f'N{r}' for r in ln['refs']] + [';']
    return ' '.join(out + ['QED'])


# ---- canonical atom order ---------------------------------------------------------------
# Atoms are renamed so that, reading the prompt left to right, they first appear as P, Q, R, S.
# The map is fixed by the prompt alone (every atom in a proof also occurs in the prompt), so
# prove.py can rename the incoming prompt, run the model, and undo the renaming on the output.
# Renaming happens on formula tuples or on prompt / compact tokens, never on spec text, where
# the reiteration rule is also spelled `R`.
ATOMS = ['P', 'Q', 'R', 'S']


def atoms_in_order(formulas):
    """Atoms by first occurrence across the given formula tuples."""
    out = []

    def walk(f):
        if f == BOT:
            return
        if f[0] == 'atom':
            if f[1] not in out:
                out.append(f[1])
            return
        for x in f[1:]:
            walk(x)
    for f in formulas:
        walk(f)
    return out


def canonical_map(premises, concl):
    """{original atom: canonical atom} for this theorem."""
    return dict(zip(atoms_in_order(premises + [concl]), ATOMS))


def rename_formula(f, m):
    if f == BOT:
        return f
    if f[0] == 'atom':
        return ('atom', m.get(f[1], f[1]))
    return (f[0],) + tuple(rename_formula(x, m) for x in f[1:])


def rename_tokens(toks, m):
    """Rename atoms in prompt or compact-body tokens (no rule is spelled like an atom there)."""
    return [m.get(t, t) for t in toks]


def canonicalize_prompt(prompt):
    """Prompt text -> (canonical prompt text, inverse map for undoing it on model output)."""
    toks = prompt.split()
    premises, concl, _ = parse_prompt(toks)
    m = canonical_map(premises, concl)
    return ' '.join(rename_tokens(toks, m)), {v: k for k, v in m.items()}


def decode_safe(prompt, body):
    """decode, but on malformed input return the raw compact text (verifier rejects it)."""
    try:
        return decode(prompt, body)
    except (ParseError, IndexError):
        return body if isinstance(body, str) else ' '.join(body)


def to_ids(toks):
    return [STOI[t] for t in toks]


def to_toks(ids):
    return [VOCAB[i] for i in ids]
