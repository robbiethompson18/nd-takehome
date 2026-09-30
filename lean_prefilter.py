"""Sound, reject-only pre-filter for `lean_seq` texts (run lean-prefilter, proposal 15 item 3).

    reject_reason(statement, text) -> None | str
        statement = LeanTokenizer.statement(prompt)   'theorem t ( P Q R S : Prop ) ( h1 : F1 ) ... : C := by'
        text      = LeanTokenizer.last_text            the literal sampled tactic block, one line
    None means "pass: Lean decides".  A string is a reason code: Lean is certain to report an error on this text.

It is a type checker for the one fragment the strict `lean_seq` grammar (`lean_tok.inverse`) admits, run only on texts
that grammar already parsed.  In that fragment every hypothesis has a declared type, the only free symbols are the
four `Prop` variables, and the only definitional unfolding Lean can do is `Not A := A -> False` -- so two formulas
are defeq iff they are syntactically equal after rewriting every `¬ A` to `A -> False` (`norm`).  Lean reports an
error for a `have` whose term does not have (up to that defeq) the declared type, and the gate rejects a theorem with
any error, so one certain error anywhere rejects the text.

What Lean accepts beyond the ND rules, and this filter therefore accepts too (see LEAN_GATE.md, soundness table):
  * `¬ A` and `A -> False` interchangeable everywhere (declared types, binders, application, `exact`);
  * `n.elim` resolved by the *declared* head of n's type (generalised field notation):
      False -> `False.elim`, any type;  `¬ A` -> `Not.elim`, type `A -> c` for any c;
      `A ∧ B` -> `And.elim`, `(A -> B -> c) -> c`;  `A ∨ B` -> `Or.elim`, `(A -> c) -> (B -> c) -> c`;
    on a declared `A -> B` (`Function.elim`) or an atom Lean errors;
  * `n.1` / `n.2` only on a declared `∧` (on `¬`, `->`, `∨`, `False`, atoms Lean errors: "invalid projection");
  * `⟨a, b⟩` only against an expected `∧`; `f x` only for an `f` whose type is `->` or `¬` (else "function expected").
Each of those error cases was checked in Lean 4.34 (tests/test_lean_prefilter.py and the edge corpus).

Anything the checker does not model -- a formula that is not in the fully parenthesised form `lean_tok.ftoks` writes
(Lean's precedence parse could differ from a naive one), a token sequence outside the grammar -- returns None (pass).
"""
import functools

ATOMS = frozenset(('P', 'Q', 'R', 'S'))
BOT = ('bot',)
OPS = {'∧': 'and', '∨': 'or', '→': 'imp'}


class _Pass(Exception):
    """not modelled: let Lean decide"""


class _Reject(Exception):
    pass


@functools.lru_cache(maxsize=1 << 18)
def norm(f):
    t = f[0]
    if t == 'atom' or t == 'bot':
        return f
    if t == 'not':
        return ('imp', norm(f[1]), BOT)
    return (t, norm(f[1]), norm(f[2]))


def _split(text):
    out = []
    for w in text.split():
        if w[0] in 'nh' and '.' in w:
            j = w.index('.')
            if j > 1 and w[1:j].isdigit() and w[j:] in ('.elim', '.1', '.2'):
                out.append(w[:j]); out.append(w[j:]); continue
        out.append(w)
    return out


class _P:
    __slots__ = ('t', 'i')

    def __init__(self, toks):
        self.t = toks; self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        t = self.peek()
        if t is None or (x is not None and t != x):
            raise _Pass('grammar')
        self.i += 1
        return t

    def formula(self):
        """the fully parenthesised form only (lean_tok.ftoks); anything else is not modelled"""
        t = self.eat()
        if t in ATOMS:
            return ('atom', t)
        if t == 'False':
            return BOT
        if t != '(':
            raise _Pass('noncanon formula')
        if self.peek() == '¬':
            self.i += 1
            a = self.formula()
            if self.eat() != ')':
                raise _Pass('noncanon formula')
            return ('not', a)
        a = self.formula()
        op = self.eat()
        if op not in OPS:
            raise _Pass('noncanon formula')
        b = self.formula()
        if self.eat() != ')':
            raise _Pass('noncanon formula')
        return (OPS[op], a, b)


@functools.lru_cache(maxsize=1 << 16)
def parse_statement(statement):
    """-> (list of premise formulas, conclusion)"""
    p = _P(statement.split())
    for x in ('theorem', 't', '(', 'P', 'Q', 'R', 'S', ':', 'Prop', ')'):
        p.eat(x)
    prem = []
    while p.peek() == '(':
        p.eat('('); h = p.eat()
        if h != f'h{len(prem) + 1}':
            raise _Pass('statement')
        p.eat(':'); prem.append(p.formula()); p.eat(')')
    p.eat(':'); concl = p.formula(); p.eat(':='); p.eat('by')
    if p.peek() is not None:
        raise _Pass('statement')
    return prem, concl


def _eq(a, b):
    return norm(a) == norm(b)


def _name(p):
    t = p.eat()
    if not (len(t) > 1 and t[0] == 'n' and t[1:].isdigit()):
        raise _Pass('grammar')
    return t


def _ref(p, scope):
    t = _name(p)
    if t not in scope:
        raise _Reject('unbound')          # lean_tok.inverse already rejects these; kept for standalone use
    return scope[t]


def _elim_ok(decl, want):
    """`n.elim : want` for n declared `decl` (generalised field notation resolves on the declared head)."""
    h = decl[0]
    w = norm(want)
    if h == 'bot':
        return True
    if h == 'not':                        # Not.elim : ¬a -> a -> α
        return w[0] == 'imp' and w[1] == norm(decl[1])
    if h == 'and':                        # And.elim : a ∧ b -> (a -> b -> α) -> α
        if w[0] != 'imp' or w[1][0] != 'imp' or w[1][2][0] != 'imp':
            return False
        f = w[1]
        return f[1] == norm(decl[1]) and f[2][1] == norm(decl[2]) and f[2][2] == w[2]
    if h == 'or':                         # Or.elim : a ∨ b -> (a -> α) -> (b -> α) -> α
        if w[0] != 'imp' or w[1][0] != 'imp' or w[2][0] != 'imp' or w[2][1][0] != 'imp':
            return False
        c = w[2][2]
        return w[1][1] == norm(decl[1]) and w[1][2] == c and w[2][1][1] == norm(decl[2]) and w[2][1][2] == c
    return False                          # `->` (Function.elim does not exist) or an atom: invalid field


def _box(p, scope, want, prem):
    """'( fun ( n : A ) => by stmts exact .. )' checked against the expected type `want`."""
    p.eat('('); p.eat('fun'); p.eat('('); nm = _name(p); p.eat(':'); a = p.formula(); p.eat(')'); p.eat('=>'); p.eat('by')
    w = norm(want)
    err = None
    if w[0] != 'imp':
        err = 'fun-not-pi'
    elif w[1] != norm(a):
        err = 'fun-binder'
    sc = dict(scope); sc[nm] = a
    _stmts(p, sc, prem)
    p.eat('exact')
    if p.peek() == '(':
        p.eat('('); e = _ref(p, sc); p.eat(':'); p.eat('False'); p.eat(')')
        if norm(e) != BOT:
            err = err or 'ascription'
        got = BOT
    else:
        got = _ref(p, sc)
    p.eat(')')
    if err is None and norm(got) != w[2]:
        err = 'fun-body'
    if err:
        raise _Reject(err)


def _term(p, scope, f, prem):
    """check the term after `have n : f :=`; raise _Reject on a certain Lean error"""
    t = p.peek()
    if t is None:
        raise _Pass('grammar')
    if t[0] == 'h' and t[1:].isdigit():
        p.eat(); k = int(t[1:])
        if k < 1 or k > len(prem):
            raise _Reject('unknown-premise')
        if not _eq(prem[k - 1], f):
            raise _Reject('premise-type')
        return
    if t == '⟨':
        p.eat(); a = _ref(p, scope); p.eat(','); b = _ref(p, scope); p.eat('⟩')
        w = norm(f)
        if w[0] != 'and':
            raise _Reject('anon-ctor')
        if norm(a) != w[1] or norm(b) != w[2]:
            raise _Reject('and-intro')
        return
    if t in ('Or.inl', 'Or.inr'):
        p.eat(); a = _ref(p, scope)
        w = norm(f)
        if w[0] != 'or':
            raise _Reject('or-intro-target')
        if norm(a) != w[1 if t == 'Or.inl' else 2]:
            raise _Reject('or-intro')
        return
    if t == 'Classical.byContradiction':
        p.eat(); p.eat('('); p.eat('fun'); p.eat('hh'); p.eat('=>'); a = _ref(p, scope); p.eat('hh'); p.eat(')')
        if norm(a) != ('imp', ('imp', norm(f), BOT), BOT):
            raise _Reject('by-contradiction')
        return
    if t == '(':
        _box(p, scope, f, prem)
        return
    if t == 'False.elim':                 # this branch's BOTE surface (`False.elim nA`, lean_tok's 2026-09-22 fix): False.elim : False -> C
        p.eat(); a = _ref(p, scope)
        if norm(a) != BOT:
            raise _Reject('false-elim')
        return
    if t == 'Or.elim':
        p.eat(); j = _ref(p, scope)
        wj = norm(j)
        if wj[0] != 'or':
            raise _Reject('or-elim-major')
        _box(p, scope, ('imp', wj[1], f), prem)
        _box(p, scope, ('imp', wj[2], f), prem)
        return
    a = _ref(p, scope)
    t2 = p.peek()
    if t2 in ('.1', '.2'):
        p.eat()
        if a[0] != 'and':
            raise _Reject('projection')
        if not _eq(a[1 if t2 == '.1' else 2], f):
            raise _Reject('and-elim')
        return
    if t2 == '.elim':
        p.eat()
        if not _elim_ok(a, f):
            raise _Reject('elim')
        return
    if t2 == ';':
        if not _eq(a, f):
            raise _Reject('reiterate')
        return
    b = _ref(p, scope)
    wa = norm(a)
    if wa[0] != 'imp':
        raise _Reject('function-expected')
    if norm(b) != wa[1]:
        raise _Reject('app-arg')
    if wa[2] != norm(f):
        raise _Reject('app-result')


def _stmts(p, scope, prem):
    while p.peek() == 'have':
        p.eat('have'); nm = _name(p); p.eat(':'); f = p.formula(); p.eat(':=')
        _term(p, scope, f, prem)
        p.eat(';')
        scope[nm] = f


def reject_reason(statement, text):
    try:
        prem, concl = parse_statement(statement)
        p = _P(_split(text))
        scope = {}
        _stmts(p, scope, prem)
        p.eat('exact')
        e = _ref(p, scope)
        if p.peek() is not None:
            raise _Pass('grammar')
        if not _eq(e, concl):
            return 'goal'
        return None
    except _Reject as r:
        return str(r)
    except (_Pass, RecursionError):
        return None
