"""Round-trip every provided proof through ndtok: spec -> compact -> spec.

    uv run python test_ndtok.py
"""
import json
from nd_verify import verify_text
from nd_verify.verify import parse_proof_tokens
from ndtok import STOI, VOCAB, decode, decode_safe, encode, parse_prompt


def provided_proofs():
    for l in open('examples/proofs_2_to_8.txt'):
        if l.startswith('THM'):
            yield 'example', l.strip()
    for l in open('targets/validation_36_reference_proofs.jsonl'):
        r = json.loads(l)
        yield r['name'], r['prompt'] + ' ' + r['reference_proof']


def skeleton(text):
    toks = text.split()
    _, _, i = parse_prompt(toks)
    return [(ln['depth'], ln['formula'], ln['rule']) for ln in parse_proof_tokens(toks[i:])]


def main():
    n = 0
    spec_len = comp_len = 0
    for name, text in provided_proofs():
        ok, reason, n_lines = verify_text(text)
        assert ok, (name, reason)
        prompt, body = encode(text)
        assert all(t in STOI for t in prompt + body), (name, [t for t in prompt + body if t not in STOI])
        back = decode(prompt, body)
        ok2, reason2, n2 = verify_text(' '.join(prompt) + ' ' + back)
        assert ok2, (name, reason2, back)
        assert n2 == n_lines, (name, n2, n_lines)
        assert skeleton(text) == skeleton(' '.join(prompt) + ' ' + back), name
        n += 1
        spec_len += len(text.split()) - len(prompt)
        comp_len += len(body)
    print(f'{n} proofs round-trip OK; body tokens spec {spec_len} -> compact {comp_len}')

    # degenerate: conclusion is a premise, proof is PR only
    assert verify_text('THM P SEQ P PRF ' + decode('THM P SEQ P PRF', 'QED'))[0]
    # malformed model output never raises, and is rejected
    for bad in ['', 'P : IMPE', 'P : R ; QED', 'Q : ANDE1 ( Q & R ) ; QED']:
        out = decode_safe('THM SEQ ( P > P ) PRF', bad)
        assert not verify_text('THM SEQ ( P > P ) PRF ' + out)[0], bad
    # duplicate boxes in ORE (S v S |- S) resolve to the same box, verifier accepts
    out = decode('THM ( S v S ) SEQ S PRF', '| S : AS ; | S : AS ; S : ORE ( S v S ) ; QED')
    assert verify_text('THM ( S v S ) SEQ S PRF ' + out)[0], out
    # the atom R and the reiteration rule get distinct model tokens (rule is REIT)
    assert len(set(VOCAB)) == len(VOCAB), 'duplicate token in VOCAB'
    _, body = encode('THM P SEQ ( Q > P ) PRF N1 P : PR ; N2 | Q : AS ; N3 | P : R N1 ; '
                     'N4 ( Q > P ) : IMPI N2 N3 ; QED')
    assert 'REIT' in body and 'R' not in body, body   # 'R' here could only be the rule
    # witness must pick the right line when two in-scope implications share a consequent
    out = decode('THM ( P > Q ) , ( R > Q ) , R SEQ Q PRF', 'Q : IMPE ( R > Q ) ; QED')
    assert out.endswith('N4 Q : IMPE N2 N3 ; QED'), out
    print('edge cases OK')

    # canonicalisation: renaming is fixed by the prompt, applies to compact tokens, and inverts
    from ndtok import canonicalize_prompt, rename_tokens
    raw = 'THM ( S > R ) , S SEQ R PRF'
    canon, inv = canonicalize_prompt(raw)
    assert canon == 'THM ( P > Q ) , P SEQ Q PRF', canon
    model_out = 'Q : IMPE ( P > Q ) ; QED'.split()
    back = decode(raw, rename_tokens(model_out, inv))
    assert verify_text(raw + ' ' + back)[0], back
    assert 'R : IMPE' in back and 'S :' in back, back      # rule REIT never collides with atom R
    canon2, inv2 = canonicalize_prompt('THM R SEQ R PRF')   # atom R only
    assert canon2 == 'THM P SEQ P PRF' and inv2 == {'P': 'R'}
    back = decode('THM R SEQ R PRF', rename_tokens('P : REIT ; QED'.split(), inv2))
    assert back == 'N1 R : PR ; N2 R : R N1 ; QED', back
    print('canonicalisation OK')

    # pruner: padded proof loses exactly its dead lines, lean proofs are untouched
    from prune import prune
    padded = ('N1 ( P > Q ) : PR ; N2 P : PR ; N3 ( P v P ) : ORI1 N2 ; N4 | S : AS ; N5 | ( S & P ) : ANDI N4 N2 ; '
              'N6 ( S > ( S & P ) ) : IMPI N4 N5 ; N7 Q : IMPE N1 N2 ; QED')
    r = prune('THM ( P > Q ) , P SEQ Q PRF', padded)
    assert (r['written'], r['pruned'], r['dead']) == (7, 3, 4), r
    assert r['proof'] == 'N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED', r['proof']
    # dead line *inside* a live box, and a dead PR-only citation chain
    padded = ('N1 ( ~ P ) : PR ; N2 | P : AS ; N3 | ( P v Q ) : ORI1 N2 ; N4 | F : NEGE N2 N1 ; '
              'N5 ( ~ P ) : NEGI N2 N4 ; QED')
    r = prune('THM ( ~ P ) SEQ ( ~ P ) PRF', padded)
    assert (r['written'], r['pruned'], r['dead']) == (5, 4, 1), r
    for name, text in provided_proofs():
        prompt, _ = encode(text)
        body = text[len(' '.join(prompt)) + 1:]
        r = prune(' '.join(prompt), body)
        assert not r.get('note'), (name, r)
    print('pruner OK')


if __name__ == '__main__':
    main()
