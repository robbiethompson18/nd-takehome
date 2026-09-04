"""Backward (goal-directed) generator of lean natural-deduction proofs.

Start from a random conclusion and recurse on what a rule would need.  Two modes per goal:

  intro : decompose the goal by its main connective (ANDI ORI1 ORI2 IMPI NEGI; NEGE for F)
  elim  : pick an anchor formula that contains the goal along an elimination path
          (ANDE1 ANDE2 IMPE DN), either an in-scope hypothesis/premise or a fresh premise
          built by wrapping the goal, and emit the chain.  ORE and BOTE are the two
          eliminations whose source does not contain the goal; handled separately.

Leanness, enforced by construction or by rejecting the sample (`Detour`):
  * eliminations only ever start from a leaf (Prawitz normal form): no intro-then-elim
  * a goal never equals an ancestor goal; a fresh premise never equals an open goal
  * a formula is never derived twice in the same scope
  * ORE / NEGI boxes must use their hypothesis; IMPI boxes may be vacuous with p_vacuous
  * `R` appears only where Fitch needs it: a box whose goal was already in scope, or a
    conclusion that is a premise (the deliberate `p_trivial` fraction)

`budget` is a target line count (premise lines included).  Leaves fire only when it is
spent, so long budgets give long proofs; the verifier's count decides the bucket.

Atoms are renamed to canonical order (ndtok.canonical_map) on the formula tuples, then the
lines are serialised in the model-facing compact form, decoded with ndtok, and verified.
"""
import argparse
import collections
import itertools
import json
import random
import time

from nd_verify import verify_text
from nd_verify.verify import BOT, parse_proof_tokens
from ndtok import ATOMS, RULE_TOK, canonical_map, decode, encode, fmt, rename_formula


class Detour(Exception):
    """The sample contains a pointless sub-derivation; regenerate."""


class Scope:
    """Formulas citable at the current point: this box's lines plus every enclosing box."""

    def __init__(self, parent=None, hyp=None):
        self.parent, self.hyp, self.hyp_used = parent, hyp, False
        self.formulas = [hyp] if hyp is not None else []

    def has(self, f):
        s = self
        while s is not None:
            if f in s.formulas:
                if f == s.hyp:
                    s.hyp_used = True
                return True
            s = s.parent
        return False

    def all(self):
        s, out = self, []
        while s is not None:
            out += s.formulas
            s = s.parent
        return out

    def hyps(self, unused=False):
        s, out = self, []
        while s is not None:
            if s.hyp is not None and not (unused and s.hyp_used):
                out.append(s.hyp)
            s = s.parent
        return out


def size(f):
    return 1 if f == BOT or f[0] == 'atom' else 1 + sum(size(x) for x in f[1:])


def elim_paths(f, G, k):
    """Chains of (rule, source formula) taking f down to G by eliminations, length <= k."""
    if f == G:
        return [[]]
    if k == 0 or f == BOT or f[0] == 'atom':
        return []
    out = []
    if f[0] == 'and':
        out += [[('ANDE1', f)] + p for p in elim_paths(f[1], G, k - 1)]
        out += [[('ANDE2', f)] + p for p in elim_paths(f[2], G, k - 1)]
    elif f[0] == 'imp':
        out += [[('IMPE', f)] + p for p in elim_paths(f[2], G, k - 1)]
    elif f[0] == 'not' and f[1][0] == 'not':
        out += [[('DN', f)] + p for p in elim_paths(f[1][1], G, k - 1)]
    return out


class Gen:
    def __init__(self, seed=0, max_fdepth=2, p_intro=0.7, p_anchor=0.7, p_hyp_minor=0.5,
                 p_ore=0.15, p_bote=0.04, p_trivial=0.005, max_chain=2, p_vacuous=0.15,
                 max_premises=3, max_premise_size=11, max_minor_size=5):
        self.rng = random.Random(seed)
        self.max_fdepth, self.p_intro, self.p_anchor = max_fdepth, p_intro, p_anchor
        self.p_hyp_minor, self.p_ore, self.p_bote = p_hyp_minor, p_ore, p_bote
        self.p_trivial, self.max_chain, self.p_vacuous = p_trivial, max_chain, p_vacuous
        self.max_premises, self.max_premise_size = max_premises, max_premise_size
        self.max_minor_size = max_minor_size
        self.detours = 0

    # ---- formulas -------------------------------------------------------------------
    def rand_formula(self, d=None, p_bot=0.0):
        r = self.rng
        if d is None:
            d = r.randint(0, self.max_fdepth)
        if r.random() < p_bot:
            return BOT
        if d == 0 or r.random() < 0.3:
            return ('atom', r.choice(ATOMS))
        k = r.choice(['not', 'not', 'and', 'or', 'imp', 'imp'])
        if k == 'not':
            return ('not', self.rand_formula(d - 1))
        return (k, self.rand_formula(d - 1), self.rand_formula(d - 1))

    def minor(self, scope, budget):
        """A formula to be proven with `budget` lines: an in-scope one (free, hypotheses
        preferred) when the budget is spent, otherwise a fresh one to recurse on."""
        r = self.rng
        unused = [f for f in scope.hyps(unused=True) if self.fresh(f)]
        hyps = [f for f in scope.hyps() if self.fresh(f)]
        pool = [f for f in scope.all() if self.fresh(f) and size(f) <= self.max_minor_size]
        if unused and r.random() < 0.8:
            return r.choice(unused)
        if hyps and r.random() < self.p_hyp_minor:
            return r.choice(hyps)
        if budget <= 0 and pool:
            return r.choice(pool)
        return self.rand_formula(1)

    def fresh(self, f):
        return f not in self.goals

    def reachable(self, f, k=2):
        """Formulas an elimination chain of length 1..k can derive from f."""
        out = []
        if k == 0 or f == BOT or f[0] == 'atom':
            return out
        if f[0] == 'and':
            out += [f[1], f[2]]
            out += self.reachable(f[1], k - 1) + self.reachable(f[2], k - 1)
        elif f[0] == 'imp':
            out += [f[2]] + self.reachable(f[2], k - 1)
        elif f[0] == 'not' and f[1][0] == 'not':
            out += [f[1][1]] + self.reachable(f[1][1], k - 1)
        return out

    def add_premise(self, M, scope):
        if M in self.goals or M in self.no_premise:
            raise Detour('premise equals an open goal')
        if scope.has(M):
            raise Detour('premise already in scope (would duplicate a hypothesis or premise)')
        if len(self.premises) >= self.max_premises or size(M) > self.max_premise_size:
            raise Detour('too many / too large premises')
        self.premises.append(M)
        self.root_scope.formulas.append(M)
        return 1

    # ---- emission -------------------------------------------------------------------
    def emit(self, depth, f, rule, witness=None):
        self.out.append((depth, f, rule, witness))


    def compact(self):
        toks = []
        for depth, f, rule, w in self.out:
            toks += (['|'] * depth + fmt(f) + [':', RULE_TOK.get(rule, rule)]
                     + (fmt(w) if w is not None else []) + [';'])
        return toks + ['QED']

    def split(self, budget, n):
        budget = max(budget, 0)
        cuts = sorted(self.rng.randint(0, budget) for _ in range(n - 1))
        return [b - a for a, b in zip([0] + cuts, cuts + [budget])]

    # ---- recursion ------------------------------------------------------------------
    def prove(self, G, depth, budget, scope, root=False):
        """Emit lines deriving G at `depth`.  Returns lines used (fresh premises included)."""
        r = self.rng
        if not root and scope.has(G):
            return 0
        if G in self.goals:
            raise Detour('goal equals an ancestor goal')
        cands = [p for f in scope.all() for p in elim_paths(f, G, self.max_chain) if p]
        if not root and G != BOT and budget <= 1:
            # out of budget: a short chain from something in scope beats a new premise
            short = [p for p in cands if len(p) == 1]
            if short and r.random() < self.p_anchor:
                return self.chain(G, r.choice(short), depth, budget, scope, 0)
            return self.add_premise(G, scope)
        self.goals.append(G)
        try:
            if G == BOT:
                # falsum only ever comes from NEGE or an in-scope anchor such as ( P > F )
                if cands and r.random() < self.p_anchor:
                    return self.chain(G, r.choice(cands), depth, budget, scope, 0)
                return self.intro(G, depth, budget, scope)
            x = r.random()
            if x < self.p_ore and budget >= 4:
                return self.ore(G, depth, budget, scope)
            if x < self.p_ore + self.p_bote:
                n = self.prove(BOT, depth, budget - 1, scope)
                self.emit(depth, G, 'BOTE')
                scope.formulas.append(G)
                return n + 1
            if G[0] != 'atom' and r.random() < self.p_intro:
                return self.intro(G, depth, budget, scope)
            return self.elim(G, depth, budget, scope, cands)
        finally:
            self.goals.pop()

    def box(self, hyp, goal, depth, budget, scope, vacuous_ok=0.0):
        """Open a box assuming hyp, prove goal inside, make sure the box ends on goal."""
        self.emit(depth + 1, hyp, 'AS')
        inner = Scope(scope, hyp)
        n = self.prove(goal, depth + 1, budget, inner)
        if self.out[-1][:2] != (depth + 1, goal):
            self.emit(depth + 1, goal, 'R')
            n += 1
        if not inner.hyp_used and self.rng.random() >= vacuous_ok:
            raise Detour('box hypothesis unused')
        self.stats['boxes'] += 1
        self.stats['hyp_used'] += inner.hyp_used
        return n + 1

    def intro(self, G, depth, budget, scope):
        r = self.rng
        if G == BOT:
            # modus-tollens shape: one side of the contradiction is in scope or a premise,
            # the other is derived (ideally from the box hypothesis)
            usable = lambda f: self.fresh(f) and ('not', f) not in self.goals
            pool = [f for f in scope.all() if usable(f)]
            negs = [f[1] for f in pool if f[0] == 'not' and usable(f[1])]
            hyps = [f for f in scope.hyps() if usable(f)]
            # things the unused hypotheses can produce by elimination: contradict one of those
            from_hyp = [x for h in scope.hyps(unused=True) for x in self.reachable(h) if usable(x)]
            n = 0
            if negs and r.random() < 0.5:
                A = r.choice(negs)                      # ( ~ A ) in scope, derive A
            elif hyps and r.random() < 0.3:
                A = r.choice(hyps)                      # A in scope, derive ( ~ A )
            else:
                A = r.choice(from_hyp) if from_hyp and r.random() < 0.7 else self.rand_formula(1)
                n = self.add_premise(('not', A), scope)  # ( ~ A ) is a premise, derive A
            n += self.prove(A, depth, budget - 1 - n, scope)
            n += self.prove(('not', A), depth, budget - 1 - n, scope)
            self.emit(depth, BOT, 'NEGE', ('not', A))
        elif G[0] == 'and':
            b1, b2 = self.split(budget - 1, 2)
            n = self.prove(G[1], depth, b1, scope)
            n += self.prove(G[2], depth, b2 + b1 - n, scope)
            self.emit(depth, G, 'ANDI')
        elif G[0] == 'or':
            side = r.choice([1, 2])
            n = self.prove(G[side], depth, budget - 1, scope)
            self.emit(depth, G, 'ORI1' if side == 1 else 'ORI2')
        elif G[0] == 'imp':
            n = self.box(G[1], G[2], depth, budget - 2, scope, vacuous_ok=self.p_vacuous)
            self.emit(depth, G, 'IMPI')
        elif G[0] == 'not':
            n = self.box(G[1], BOT, depth, budget - 2, scope)
            self.emit(depth, G, 'NEGI')
        scope.formulas.append(G)
        return n + 1

    def elim(self, G, depth, budget, scope, cands):
        r = self.rng
        if budget >= 4:                     # with budget to spend, prefer chains that have a minor
            cands = [p for p in cands if any(rule == 'IMPE' for rule, _ in p)] or cands
        if cands and r.random() < self.p_anchor:
            return self.chain(G, r.choice(cands), depth, budget, scope, 0)
        if G[0] != 'atom' and r.random() < 0.8:
            return self.intro(G, depth, budget, scope)
        # fresh premise: wrap G k times, record the path back down, spend the budget on minors
        k = max(1, min(1 + int(r.expovariate(1.0)), self.max_chain, budget - 1))
        rules = [r.choice(['ANDE1', 'ANDE2', 'IMPE', 'IMPE', 'DN']) for _ in range(k)]
        if 'IMPE' not in rules and (budget - 1 - k >= 2 or scope.hyps(unused=True)):
            rules[r.randrange(k)] = 'IMPE'      # only a minor can absorb budget or use a hypothesis
        minors = self.split(budget - 1 - k, sum(w == 'IMPE' for w in rules) or 1)
        M, chain = G, []
        for w in rules:
            if w == 'IMPE':
                M = ('imp', self.minor(scope, minors.pop()), M)
            elif w == 'ANDE1':
                M = ('and', M, self.minor(scope, 0))
            elif w == 'ANDE2':
                M = ('and', self.minor(scope, 0), M)
            elif M[0] == 'not' and M[1][0] == 'not':
                continue
            else:
                M = ('not', ('not', M))
            chain.insert(0, (w, M))
        n = 0 if scope.has(M) else self.add_premise(M, scope)
        return self.chain(G, chain, depth, budget, scope, n)

    def chain(self, G, chain, depth, budget, scope, n):
        """Emit an elimination chain top-down; the chain's source is already in scope."""
        minors = [i for i, (rule, _) in enumerate(chain) if rule == 'IMPE']
        mbudget = dict(zip(minors, self.split(budget - n - len(chain), len(minors)))) if minors else {}
        for i, (rule, src) in enumerate(chain):
            derived = chain[i + 1][1] if i + 1 < len(chain) else G
            if scope.has(G):
                break                      # target reached while proving a minor
            if scope.has(derived):
                continue
            if rule == 'IMPE':
                n += self.prove(src[1], depth, mbudget[i], scope)
            self.emit(depth, derived, rule, src if rule != 'DN' else None)
            scope.formulas.append(derived)
            n += 1
        return n

    def ore(self, G, depth, budget, scope):
        r = self.rng
        disj = [f for f in scope.all() if f[0] == 'or']
        n = 0
        if disj and r.random() < self.p_anchor:
            D = r.choice(disj)
        else:
            A = G if r.random() < 0.3 else self.rand_formula(1)   # one disjunct may be the goal
            B = self.rand_formula(1)
            if A == B:
                B = self.rand_formula(1)
            D = ('or', A, B) if r.random() < 0.5 else ('or', B, A)
            if not scope.has(D):
                n = self.add_premise(D, scope)
        if D[1] == D[2]:
            raise Detour('ORE on ( A v A ): identical boxes, one would be dead')
        b1, b2 = self.split(budget - n - 3, 2)
        # the boxes re-prove G under a hypothesis: not a cycle, but G must not become a premise
        self.goals.remove(G)
        self.no_premise.append(G)
        try:
            n += self.box(D[1], G, depth, b1, scope)
            n += self.box(D[2], G, depth, b2, scope)
        finally:
            self.goals.append(G)
            self.no_premise.pop()
        self.emit(depth, G, 'ORE', D)
        scope.formulas.append(G)
        return n + 1

    # ---- top level ------------------------------------------------------------------
    def nested_root(self):
        """A conclusion whose introduction opens a box inside a box: ( A > ( B > C ) ),
        ( A > ( ~ B ) ), ( ~ ( A > B ) ), ( ~ ( ~ A ) ) ... two box-introducing connectives stacked."""
        r = self.rng
        inner = ('imp', self.rand_formula(1), self.rand_formula(1)) if r.random() < 0.6 \
            else ('not', self.rand_formula(1))
        return ('imp', self.rand_formula(1), inner) if r.random() < 0.6 else ('not', inner)

    def generate(self, budget, root=None):
        """One verified, canonical record.  Raises if the verifier rejects (a generator bug).
        `root` fixes the conclusion (used by nested mode); otherwise it is random."""
        while True:
            self.out, self.premises, self.goals, self.no_premise = [], [], [], []
            self.root_scope = Scope()
            self.stats = collections.Counter()
            G = root if root is not None else self.rand_formula(p_bot=0.02)
            try:
                if root is None and self.rng.random() < self.p_trivial:
                    self.premises.append(G)
                    self.emit(0, G, 'R')
                else:
                    self.prove(G, 0, budget, self.root_scope, root=True)
                    if self.out[-1][:2] != (0, G):
                        raise Detour('conclusion derived before the end')
                # canonical atom names, applied to the formulas before anything is serialised
                m = canonical_map(self.premises, G)
                self.premises = [rename_formula(p, m) for p in self.premises]
                G = rename_formula(G, m)
                self.out = [(d, rename_formula(f, m), rule, rename_formula(w, m) if w is not None else None)
                            for d, f, rule, w in self.out]
                prompt = 'THM ' + (' , '.join(' '.join(fmt(p)) for p in self.premises) + ' '
                                   if self.premises else '') + 'SEQ ' + ' '.join(fmt(G)) + ' PRF'
                body = decode(prompt, self.compact())
                # a premise the decoded proof never cites (e.g. it duplicates a box hypothesis,
                # which the decoder prefers) makes a padded theorem: reject
                lines = parse_proof_tokens(body.split())
                cited = {r for ln in lines for r in ln['refs']}
                if any(ln['rule'] == 'PR' and ln['idx'] not in cited for ln in lines):
                    raise Detour('a premise is never cited')
                break
            except Detour:
                self.detours += 1
                if root is not None:
                    G = root
        text = prompt + ' ' + body
        ok, reason, n_lines = verify_text(text)
        if not ok:
            raise RuntimeError(f'generator emitted an invalid proof: {reason}\n{text}')
        ptoks, body = encode(text)
        i = ptoks.index('SEQ')
        return {'thm': ' '.join(ptoks[1:i]) + ' |- ' + ' '.join(ptoks[i + 1:-1]),
                'key': theorem_key(self.premises, G),
                'prompt': ' '.join(ptoks), 'proof': text[len(' '.join(ptoks)) + 1:],
                'n_lines': n_lines, 'n_premises': len(self.premises),
                'rules': [t for i, t in enumerate(body) if i > 0 and body[i - 1] == ':'],
                'max_depth': max([b.count('|') for b in ' '.join(body).split(';')] + [0]),
                'boxes': self.stats['boxes'], 'hyp_used': self.stats['hyp_used'],
                'trivial': trivial_kind(self.premises, G)}


def theorem_key(premises, G):
    """Identity of a theorem up to atom renaming and premise order: the smallest canonical
    string over all premise permutations.  Two records with the same key are duplicates."""
    best = None
    for perm in itertools.permutations(premises):
        m = canonical_map(list(perm), G)
        s = ' , '.join(' '.join(fmt(rename_formula(p, m))) for p in perm) \
            + ' |- ' + ' '.join(fmt(rename_formula(G, m)))
        best = s if best is None or s < best else best
    return best


def trivial_kind(premises, G):
    if G in premises:
        return 'conclusion_is_premise'
    if BOT in premises:
        return 'falsum_premise'
    if G != BOT and G[0] in ('and', 'or') and G[1] == G[2] and G[1] in premises:
        return 'a_gives_a_op_a'
    return None


def sample(per_len, lo, hi, seed=0, banned=frozenset(), trivial_frac=0.01,
           max_tries=2_000_000, seconds=None, knobs=None, nested=False):
    """Fill length buckets lo..hi with up to `per_len` distinct theorems each.

    Distinct means distinct `key` (atom renaming and premise order folded).  Theorems whose key
    is in `banned` are skipped (used to keep everything in targets/ out of training).  Trivial
    theorems are capped at `trivial_frac` of a bucket.  Stops when every bucket is full, or at
    `max_tries` attempts or `seconds` wall-clock, whichever comes first -- short buckets mean the
    generator ran out of distinct theorems at that length.  Returns ({L: records}, stats).

    nested=True: conclusions are stacked box-introductions (see Gen.nested_root), introduction
    is always preferred, and only proofs reaching box depth >= 2 are kept.  Under the 6-line cap
    such proofs are rare by chance (~0.2%), so they are sampled separately and merged."""
    g = Gen(seed=seed, **{**(knobs or {}), **({'p_intro': 1.0} if nested else {})})
    want = {L: per_len for L in range(lo, hi + 1)}
    got = collections.defaultdict(list)
    n_triv, seen = collections.Counter(), set()
    st = collections.Counter()
    t0 = time.time()
    while any(len(got[L]) < want[L] for L in want) and st['tries'] < max_tries \
            and (seconds is None or time.time() - t0 < seconds):
        st['tries'] += 1
        rec = g.generate(budget=g.rng.randint(lo, hi + 3), root=g.nested_root() if nested else None)
        L = rec['n_lines']
        if L not in want or len(got[L]) >= want[L] or (nested and rec['max_depth'] < 2):
            st['out_of_range_or_full'] += 1
            continue
        if rec['key'] in seen:
            st['dupes'] += 1
            continue
        if rec['key'] in banned:
            st['banned'] += 1
            continue
        if rec['trivial'] and n_triv[L] >= max(1, int(trivial_frac * per_len)):
            st['trivial_capped'] += 1
            continue
        seen.add(rec['key'])
        got[L].append(rec)
        n_triv[L] += bool(rec['trivial'])
    st['detours'] = g.detours
    st['seconds'] = round(time.time() - t0, 1)
    return got, st


def describe(got, st, show=0):
    total = sum(len(v) for v in got.values())
    print(f'{total} records in {st["seconds"]}s: {dict(st)}')
    for L in sorted(got):
        recs = got[L]
        rules = collections.Counter(r for x in recs for r in x['rules'])
        triv = sum(1 for x in recs if x['trivial'])
        prem = sum(x['n_premises'] for x in recs) / max(len(recs), 1)
        boxes, used = sum(x['boxes'] for x in recs), sum(x['hyp_used'] for x in recs)
        depth = collections.Counter(x['max_depth'] for x in recs)
        print(f'\n== length {L}: {len(recs)} recs, trivial {triv}, mean premises {prem:.1f}, '
              f'boxes {boxes} (hyp used {used}), depth {dict(sorted(depth.items()))}, '
              f'rules {dict(rules.most_common(8))}')
        for x in recs[:show]:
            print('  ', x['thm'], '   ::', x['proof'])


def main():
    ap = argparse.ArgumentParser(description='inspect the generator; make_data.py builds the real splits')
    ap.add_argument('--n', type=int, default=10, help='records per length bucket')
    ap.add_argument('--lo', type=int, default=2)
    ap.add_argument('--hi', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--show', type=int, default=2, help='print this many examples per bucket')
    ap.add_argument('--max-tries', type=int, default=2_000_000)
    ap.add_argument('--nested', action='store_true', help='only depth >= 2 proofs (see sample)')
    a = ap.parse_args()
    got, st = sample(a.n, a.lo, a.hi, seed=a.seed, max_tries=a.max_tries, nested=a.nested)
    describe(got, st, a.show)
    if a.out:
        with open(a.out, 'w') as f:
            for L in sorted(got):
                for x in got[L]:
                    f.write(json.dumps(x) + '\n')
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
