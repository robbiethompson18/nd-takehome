"""Lean as the in-loop checker for Lean-format models (run lean-format).

gate(tok, prompts, nd_proofs, texts) is called by sample.generate() for a LeanTokenizer.  For every distinct
(prompt, literal Lean text) whose text parsed in the strict grammar it (i) asks Lean (core) whether the literal text proves
the theorem, (ii) asks nd_verify whether the denoted ND proof does, logs the 2 x 2 table and the timings, writes every
disagreement to <log>.disagree.jsonl, and returns the ND proofs, prefixing 'LEANREJ ' where Lean rejects what nd_verify would accept (so that
nd_verify-based judging downstream accepts exactly the samples that BOTH checkers accept).

One theorem per line in a chunk file, `lean -DmaxErrors=...` so every error is reported; error line -> theorem.  A chunk
whose lean process crashes / times out is split recursively; a single theorem that crashes is rejected.
Log: $LEAN_GATE_LOG (default artifacts/lean_gate.jsonl), one json line per generate() call.

Speed (ported from dan-pandori/nd-takehome `dan`, run lean-prefilter 2026-09-28): lean_prefilter.reject_reason drops texts
Lean is certain to reject before Lean sees them ($LEAN_PREFILTER on|off|shadow); workers default to the cgroup CPU quota and
each Lean runs `-j $LEAN_GATE_THREADS` (1), since Lean's default of one thread per visible core oversubscribes RunPod pods.
"""
import os, re, sys, json, time, subprocess, tempfile, shutil, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text

LEAN = os.path.expanduser('~/.elan/bin/lean')
CHUNK = int(os.environ.get('LEAN_GATE_CHUNK', '400'))


def cpu_quota():
    """CPUs this process may actually use: the cgroup quota intersected with the affinity mask (RunPod pods show 96 cores
    but grant ~7.6).  Copied from dan-pandori/nd-takehome `dan` lean_gate.py (run lean-prefilter, 2026-09-28)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    for fq, fp in (('/sys/fs/cgroup/cpu.max', None),                                            # cgroup v2
                   ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')):  # v1 (RunPod A40s)
        try:
            if fp is None:
                q, per = open(fq).read().split()[:2]
            else:
                q, per = open(fq).read().strip(), open(fp).read().strip()
            if q not in ('max', '-1'):
                n = min(n, max(1, int(int(q) / int(per))))
            break
        except (OSError, ValueError):
            continue
    return n


WORKERS = int(os.environ.get('LEAN_GATE_WORKERS', '0')) or cpu_quota()
THREADS = os.environ.get('LEAN_GATE_THREADS', '1')      # `lean -j N`; '' = Lean's default (a thread per hardware thread)
PREFILTER = os.environ.get('LEAN_PREFILTER', 'on')      # on | off | shadow (lean_prefilter.py)
assert PREFILTER in ('on', 'off', 'shadow'), PREFILTER
ERR = re.compile(r'^[^\n]*?:(\d+):\d+: error', re.M)


def _run(lines, workdir, tag, depth=0):
    """lines: list of one-line theorem sources -> (list of bool, cpu seconds)"""
    fn = os.path.join(workdir, f'c_{tag}.lean')
    with open(fn, 'w') as f:
        f.write('set_option linter.unusedVariables false\n' + '\n'.join(lines) + '\n')
    t0 = time.time()
    try:
        p = subprocess.run([LEAN] + (['-j', THREADS] if THREADS else []) + ['-DmaxErrors=100000000', fn], capture_output=True, text=True, timeout=60 + 2 * len(lines))
        o = p.stdout + p.stderr; rc = p.returncode
    except subprocess.TimeoutExpired:
        o = ''; rc = -9
    cpu = time.time() - t0
    os.remove(fn)
    bad = {int(m.group(1)) - 2 for m in ERR.finditer(o)}
    crashed = rc not in (0, 1) or (rc == 1 and not bad) or any(b < 0 or b >= len(lines) for b in bad)
    if crashed:
        if len(lines) == 1:
            return [False], cpu
        h = len(lines) // 2
        a, ca = _run(lines[:h], workdir, tag + 'a', depth + 1); b, cb = _run(lines[h:], workdir, tag + 'b', depth + 1)
        return a + b, cpu + ca + cb
    return [k not in bad for k in range(len(lines))], cpu


def lean_check(items):
    """items: list of (statement text, tactic text) -> (list of bool, wall seconds, summed process seconds)"""
    if not items:
        return [], 0.0, 0.0
    lines = [f'{s.replace("theorem t ", f"theorem t{k} ", 1)} {b}' for k, (s, b) in enumerate(items)]
    wd = tempfile.mkdtemp(prefix='leangate_')
    t0 = time.time()
    chunks = [(lines[i:i + CHUNK], i) for i in range(0, len(lines), CHUNK)]
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(lambda c: _run(c[0], wd, str(c[1])), chunks))
    shutil.rmtree(wd, ignore_errors=True)
    ok = [x for r, _ in res for x in r]
    return ok, time.time() - t0, sum(c for _, c in res)


def gate(tok, prompts, nd_proofs, texts):
    logfn = os.environ.get('LEAN_GATE_LOG', 'artifacts/lean_gate.jsonl')
    keys = {}
    for p, nd, tx in zip(prompts, nd_proofs, texts):
        if tx is not None and not nd.startswith('LEANPARSE'):
            keys.setdefault((p, tx), nd)
    items = list(keys.items())
    # reject-only pre-filter: 'on' sends only the texts it passes to Lean, 'shadow' sends everything and logs any text
    # it rejects that Lean accepts (a filter bug) to <log>.filterbug.jsonl
    t0 = time.time()
    filt = [None] * len(items)
    if PREFILTER != 'off':
        from lean_prefilter import reject_reason
        filt = [reject_reason(tok.statement(p), tx) for (p, tx), _ in items]
    filter_s = time.time() - t0
    to_lean = [k for k, r in enumerate(filt) if r is None or PREFILTER == 'shadow']
    ok_sent, wall, cpu = lean_check([(tok.statement(items[k][0][0]), items[k][0][1]) for k in to_lean])
    lean_ok = [False] * len(items)
    for k, o in zip(to_lean, ok_sent):
        lean_ok[k] = o
    bugs = [k for k in range(len(items)) if filt[k] is not None and lean_ok[k]]
    if PREFILTER == 'on':
        lean_ok = [o and filt[k] is None for k, o in enumerate(lean_ok)]
    t0 = time.time()
    nd_ok = [verify_text(p + ' ' + nd)[0] for (p, tx), nd in items]
    t_nd = time.time() - t0
    verdict = {k: lo for (k, _), lo in zip(items, lean_ok)}
    tab = collections.Counter((bool(a), bool(b)) for a, b in zip(nd_ok, lean_ok))
    dis = [{'prompt': p, 'lean_text': tx, 'nd': nd, 'nd_ok': bool(a), 'lean_ok': bool(b)} for ((p, tx), nd), a, b in zip(items, nd_ok, lean_ok) if bool(a) != bool(b)]
    n_parse = sum(1 for nd in nd_proofs if nd.startswith('LEANPARSE'))
    parse_reasons = collections.Counter(nd[10:] for nd in nd_proofs if nd.startswith('LEANPARSE'))
    rec = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'samples': len(prompts), 'parse_fail': n_parse, 'distinct_checked': len(items),
           'both_ok': tab[(True, True)], 'nd_ok_lean_rej': tab[(True, False)], 'nd_rej_lean_ok': tab[(False, True)], 'both_rej': tab[(False, False)],
           'parse_reasons': dict(parse_reasons.most_common()), 'lean_wall_s': wall, 'lean_proc_s': cpu, 'nd_verify_s': t_nd, 'workers': WORKERS, 'chunk': CHUNK,
           'threads': THREADS, 'prefilter': PREFILTER, 'lean_sent': len(to_lean), 'filter_rej': sum(r is not None for r in filt),
           'filter_reasons': dict(collections.Counter(r for r in filt if r is not None).most_common()), 'filter_s': filter_s,
           'filter_false_rej': len(bugs) if PREFILTER == 'shadow' else None}
    os.makedirs(os.path.dirname(logfn) or '.', exist_ok=True)
    with open(logfn, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    if bugs:
        with open(logfn.replace('.jsonl', '') + '.filterbug.jsonl', 'a') as f:
            for k in bugs:
                (p, tx), nd = items[k]
                f.write(json.dumps({'prompt': p, 'lean_text': tx, 'filter_reason': filt[k]}, ensure_ascii=False) + '\n')
        print(f'[lean_gate] PREFILTER BUG: {len(bugs)} texts the filter rejects are accepted by Lean', flush=True)
    if dis:
        with open(logfn.replace('.jsonl', '') + '.disagree.jsonl', 'a') as f:
            for d in dis:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'[lean_gate] {len(prompts)} samples, parse-fail {n_parse}, distinct checked {len(items)}: both ok {tab[(True, True)]}, nd-only {tab[(True, False)]}, '
          f'lean-only {tab[(False, True)]}, both rej {tab[(False, False)]}; lean {wall:.1f}s wall ({cpu:.1f}s proc, {WORKERS} workers, -j {THREADS or "default"}) on {len(to_lean)} texts, '
          f'prefilter={PREFILTER} rejected {sum(r is not None for r in filt)} in {filter_s:.1f}s; nd_verify {t_nd:.1f}s', flush=True)
    ndv = {k: a for (k, _), a in zip(items, nd_ok)}
    out = []
    for p, nd, tx in zip(prompts, nd_proofs, texts):
        if tx is None or nd.startswith('LEANPARSE') or verdict[(p, tx)] or not ndv[(p, tx)]:
            out.append(nd)               # accepted by Lean, or rejected by nd_verify too (judge() then records nd_verify's reason)
        else:
            out.append('LEANREJ ' + nd)  # nd_verify would accept but Lean rejected the literal text: never counted
    return out
