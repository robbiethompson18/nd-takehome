"""lean-prefilter unit tests: the pre-filter against Lean on hand-made texts, one per rule and per unfolding Lean allows.
    python3 tests/test_lean_prefilter.py          (Lean 4 core on PATH or ~/.elan/bin/lean)
Each case: (ND prompt, lean_seq text, Lean's verdict).  The filter must never reject a case Lean accepts, and must
reject the listed rejects with the listed reason.  Lean re-decides every case, so a wrong expectation fails too."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lean_tok import LeanTokenizer
from lean_prefilter import reject_reason
from lean_gate import lean_check

A = 'THM ( P & Q ) SEQ ( ( P > ( Q > R ) ) > R ) PRF'
O = 'THM ( P v Q ) SEQ ( ( P > R ) > ( ( Q > R ) > R ) ) PRF'
N = 'THM ( ~ P ) SEQ ( P > R ) PRF'
NP = 'THM ( ~ P ) , P SEQ F PRF'
F = 'THM F SEQ P PRF'
DN = 'THM ( ~ ( ~ P ) ) SEQ P PRF'
I = 'THM P SEQ ( ~ ( ~ P ) ) PRF'
C = 'THM P , ( P > Q ) SEQ Q PRF'
CASES = [
    # accepted beyond the ND rules
    (A, 'have n1 : ( P ∧ Q ) := h1 ; have n2 : ( ( P → ( Q → R ) ) → R ) := n1.elim ; exact n2', True, None),
    (O, 'have n1 : ( P ∨ Q ) := h1 ; have n2 : ( ( P → R ) → ( ( Q → R ) → R ) ) := n1.elim ; exact n2', True, None),
    (N, 'have n1 : ( ¬ P ) := h1 ; have n2 : ( P → R ) := n1.elim ; exact n2', True, None),
    (N, 'have n1 : ( P → False ) := h1 ; have n2 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : False := n1 n3 ; have n5 : R := n4.elim ; exact n5 ) ; exact n2', True, None),
    (NP, 'have n1 : ( ¬ P ) := h1 ; have n2 : P := h2 ; have n3 : False := n1 n2 ; exact n3', True, None),
    (F, 'have n1 : False := h1 ; have n2 : P := n1.elim ; exact n2', True, None),
    (DN, 'have n1 : ( ( P → False ) → False ) := h1 ; have n2 : P := Classical.byContradiction ( fun hh => n1 hh ) ; exact n2', True, None),
    (I, 'have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : ( P → False ) ) => by have n4 : False := n3 n1 ; exact ( n4 : False ) ) ; exact n2', True, None),
    (I, 'have n1 : P := h1 ; have n2 : ( ( ¬ P ) → False ) := ( fun ( n3 : ( ¬ P ) ) => by have n4 : False := n3 n1 ; exact n4 ) ; exact n2', True, None),
    (N, 'have n1 : ( ¬ P ) := h1 ; have n2 : ( ¬ P ) := n1.elim ; have n3 : ( P → R ) := ( fun ( n4 : P ) => by have n5 : False := n2 n4 ; have n6 : R := n5.elim ; exact n6 ) ; exact n3', True, None),
    # certain Lean errors
    (N, 'have n1 : ( P → False ) := h1 ; have n2 : ( P → R ) := n1.elim ; exact n2', False, 'elim'),
    (C, 'have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : ( Q → Q ) := n1.elim ; exact n2', False, 'elim'),
    (N, 'have n1 : ( ¬ P ) := h1 ; have n2 : P := n1.1 ; exact n1', False, 'projection'),
    (O, 'have n1 : ( P ∨ Q ) := h1 ; have n2 : P := n1.1 ; exact n2', False, 'projection'),
    (F, 'have n1 : False := h1 ; have n2 : P := n1.1 ; exact n2', False, 'projection'),
    (A, 'have n1 : ( P ∧ Q ) := h1 ; have n2 : P := n1 n1 ; exact n2', False, 'function-expected'),
    (C, 'have n1 : P := h1 ; have n2 : Q := n1 n1 ; exact n2', False, 'function-expected'),
    (I, 'have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ⟨ n1 , n1 ⟩ ; exact n2', False, 'anon-ctor'),
    (O, 'have n1 : ( P ∨ Q ) := h1 ; have n2 : ( P ∨ Q ) := ⟨ n1 , n1 ⟩ ; exact n2', False, 'anon-ctor'),
    (A, 'have n1 : ( P ∧ Q ) := h1 ; have n2 : ( P ∧ Q ) := n1.elim ; exact n2', False, 'elim'),
    (O, 'have n1 : ( P ∨ Q ) := h1 ; have n2 : P := n1.elim ; exact n2', False, 'elim'),
    (I, 'have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := Or.inl n1 ; exact n2', False, 'or-intro-target'),
    (C, 'have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : Q := n2 n2 ; exact n3', False, 'app-arg'),
    (C, 'have n1 : P := h1 ; have n2 : ( P → Q ) := h2 ; have n3 : P := n2 n1 ; exact n3', False, 'app-result'),
    (C, 'have n1 : P := h1 ; have n2 : ( P → Q ) := h3 ; exact n2', False, 'unknown-premise'),
    (C, 'have n1 : Q := h1 ; exact n1', False, 'premise-type'),
    (C, 'have n1 : P := h1 ; exact n1', False, 'goal'),
    (DN, 'have n1 : ( ¬ ( ¬ P ) ) := h1 ; have n2 : Q := Classical.byContradiction ( fun hh => n1 hh ) ; exact n2', False, 'by-contradiction'),
    (I, 'have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : P ) => by exact n1 ) ; exact n2', False, 'fun-binder'),
    (C, 'have n1 : P := h1 ; have n2 : Q := ( fun ( n3 : P ) => by exact n3 ) ; exact n2', False, 'fun-not-pi'),
    (I, 'have n1 : P := h1 ; have n2 : ( ¬ ( ¬ P ) ) := ( fun ( n3 : ( ¬ P ) ) => by exact ( n1 : False ) ) ; exact n2', False, 'ascription'),
    # this branch's BOTE surface, `False.elim nA` (lean_tok 2026-09-22 fix)
    (F, 'have n1 : False := h1 ; have n2 : P := False.elim n1 ; exact n2', True, None),
    (NP, 'have n1 : ( ¬ P ) := h1 ; have n2 : P := h2 ; have n3 : False := n1 n2 ; have n4 : False := False.elim n3 ; exact n4', True, None),
    (N, 'have n1 : ( ¬ P ) := h1 ; have n2 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : False := n1 n3 ; have n5 : R := False.elim n4 ; exact n5 ) ; exact n2', True, None),
    (C, 'have n1 : P := h1 ; have n2 : Q := False.elim n1 ; exact n2', False, 'false-elim'),
    (N, 'have n1 : ( ¬ P ) := h1 ; have n2 : ( P → R ) := False.elim n1 ; exact n2', False, 'false-elim'),
    (N, 'have n1 : ( P → False ) := h1 ; have n2 : R := False.elim n1 ; exact n2', False, 'false-elim'),
]
tok = LeanTokenizer('lean_seq')
fails = 0
ok, _, _ = lean_check([(tok.statement(p), t) for p, t, _, _ in CASES])
for (p, t, want, reason), lean in zip(CASES, ok):
    got = reject_reason(tok.statement(p), t)
    good = (lean == want) and (got is None if want else got == reason)
    fails += not good
    print('PASS' if good else 'FAIL', f'lean={lean} expected={want} filter={got} want={reason} :: {t[:90]}')
print('\nLEAN PREFILTER TESTS', 'PASS' if not fails else f'FAIL ({fails})')
sys.exit(1 if fails else 0)
