"""Rent an H100, run a command on it, pull results back, terminate.

    uv run python pod.py                       # the Stage-1 run (both positional schemes)
    uv run python pod.py --cmd 'nvidia-smi'    # anything else
    uv run python pod.py --keep                # leave the pod up afterwards

The pod is terminated in a `finally`, including on Ctrl-C or a remote failure, so a crashed run
costs minutes rather than hours.  Needs RUNPOD_API_KEY (direnv, .envrc.local) and the ssh key
registered with the RunPod account.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

API = 'https://api.runpod.io/graphql'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'   # torch 2.4 + numpy, no sync
REMOTE = '/workspace/nd'
STAGE1 = f"""
set -e
cd {REMOTE}
python test_ndtok.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
wc -l data/train.jsonl data/heldout.jsonl
python train.py --pos {{pos}} --batch {{batch}} --steps {{steps}} --lr {{lr}} --seed {{seed}} \
  --eval-every 1000 --out ckpts/stage1_{{pos}}.pt 2>&1 | tee logs/stage1_{{pos}}.log
"""


REGISTRY = 'logs/pods.json'


def registry(name=None, value=...):
    """Read the warm-pod registry, or set/delete one entry and write it back."""
    reg = json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}
    if name is not None:
        reg.pop(name, None) if value is None else reg.__setitem__(name, value)
        os.makedirs(os.path.dirname(REGISTRY) or '.', exist_ok=True)
        json.dump(reg, open(REGISTRY, 'w'), indent=2)
    return reg


def gql(query, key):
    req = urllib.request.Request(f'{API}?api_key={key}', method='POST',
                                 data=json.dumps({'query': query}).encode(),
                                 # RunPod's edge 403s python-urllib's default User-Agent
                                 headers={'Content-Type': 'application/json',
                                          'User-Agent': 'curl/8.7.1'})
    out = json.load(urllib.request.urlopen(req))
    if 'errors' in out:
        raise SystemExit(json.dumps(out['errors'], indent=2))
    return out['data']


def create(key, gpu, name, pubkey):
    q = ('mutation { podFindAndDeployOnDemand(input: {'
         f'cloudType: SECURE, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 60, '
         f'minVcpuCount: 8, minMemoryInGb: 32, gpuTypeId: "{gpu}", name: "{name}", '
         f'imageName: "{IMAGE}", ports: "22/tcp", volumeMountPath: "/workspace", '
         f'supportPublicIp: true, env: [{{key: "PUBLIC_KEY", value: "{pubkey}"}}]'
         '}) { id costPerHr } }')
    pod = gql(q, key)['podFindAndDeployOnDemand']
    if pod is None:
        raise SystemExit(f'no {gpu} capacity right now')
    return pod


def wait_ssh(key, pod_id, timeout=600):
    q = ('query { pod(input: {podId: "%s"}) { desiredStatus '
         'runtime { ports { ip publicPort privatePort isIpPublic } } } }' % pod_id)
    t0 = time.time()
    while time.time() - t0 < timeout:
        rt = gql(q, key)['pod']['runtime']
        for p in (rt or {}).get('ports') or []:
            if p['privatePort'] == 22 and p['isIpPublic']:
                return p['ip'], p['publicPort']
        time.sleep(5)
    raise SystemExit('pod never exposed ssh')


def ssh_args(ip, port):
    return ['-p', str(port), '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR', f'root@{ip}']


def run(cmd, **kw):
    print(f'$ {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sh(cmd):
    """The RunPod pytorch image has no rsync, so files move as a tar stream over ssh."""
    print(f'$ {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='NVIDIA H100 80GB HBM3')
    ap.add_argument('--name', default=None)
    ap.add_argument('--cmd', default=None, help='remote shell command (default: the Stage-1 run)')
    ap.add_argument('--pos', default='nope', choices=['nope', 'rope'])
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--terminate', action='store_true',
                    help='tear the pod down when the run ends. Off by default: pods stay warm so '
                         'the next run skips the ~2 min provision, and are killed with --kill.')
    ap.add_argument('--reuse', default=None, metavar='NAME',
                    help='run on a pod already in the registry instead of creating one')
    ap.add_argument('--kill', default=None, metavar='NAME|all', help='terminate pod(s) and exit')
    ap.add_argument('--list', action='store_true', help='show live pods and exit')
    a = ap.parse_args()

    key = os.environ.get('RUNPOD_API_KEY') or sys.exit('RUNPOD_API_KEY unset (direnv allow?)')
    if a.list:
        for p in gql('query { myself { pods { id name desiredStatus costPerHr } } }',
                     key)['myself']['pods']:
            print(p)
        return
    if a.kill:
        for name, p in list(registry().items()):
            if a.kill in (name, 'all'):
                gql('mutation { podTerminate(input: {podId: "%s"}) }' % p['id'], key)
                print(f'terminated {name} ({p["id"]})')
                registry(name, None)
        return

    pubkey = open(os.path.expanduser('~/.ssh/id_ed25519.pub')).read().strip()
    cmd = a.cmd or STAGE1.format(pos=a.pos, batch=a.batch, steps=a.steps, lr=a.lr, seed=a.seed)
    name = a.name or a.reuse or f'nd-stage1-{a.pos}'
    # data generation is CPU work; do it locally (free) and ship the split up
    assert a.cmd or os.path.exists('data/train.jsonl'), 'run make_data.py first'

    t0 = time.time()
    if a.reuse:
        pod = registry()[a.reuse]
        print(f"reusing warm pod {a.reuse} ({pod['id']}) at ${pod['costPerHr']}/hr", flush=True)
    else:
        pod = create(key, a.gpu, name, pubkey)
        print(f"pod {pod['id']} at ${pod['costPerHr']}/hr", flush=True)
    try:
        ip, port = (pod['ip'], pod['port']) if a.reuse else wait_ssh(key, pod['id'])
        print(f'ssh -p {port} root@{ip}   (up after {time.time()-t0:.0f}s)', flush=True)
        registry(name, {**pod, 'ip': ip, 'port': port})
        for _ in range(30):                      # sshd starts a little after the port opens
            if subprocess.run(['ssh', *ssh_args(ip, port), 'true'], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        ssh = 'ssh ' + ' '.join(ssh_args(ip, port))
        # ckpts ships up: Stage 2 initialises from stage1_*.pt, and the frozen control
        # loads it too.  logs stay local so a reused pod does not overwrite them.
        excl = ' '.join(f'--exclude ./{d}' for d in ('.git', '.venv', 'logs', '__pycache__'))
        sh(f'tar czf - {excl} . | {ssh} "mkdir -p {REMOTE}/ckpts {REMOTE}/logs '
           f'{REMOTE}/data/harvest && tar xzf - -C {REMOTE}"')
        # shlex.quote, not json.dumps: JSON escapes newlines to literal \n, which bash then
        # reads as the two characters backslash-n rather than as line breaks.
        run(['ssh', *ssh_args(ip, port), 'bash', '-lc', shlex.quote(cmd)])
        sh(f'{ssh} "cd {REMOTE} && tar czf - ckpts logs data/harvest" | tar xzf - -C .')
    finally:
        mins = (time.time() - t0) / 60
        if not a.terminate:
            print(f'pod {name} ({pod["id"]}) LEFT RUNNING at ${pod["costPerHr"]}/hr — '
                  f'reuse with --reuse {name}, kill with --kill {name}')
        else:
            gql('mutation { podTerminate(input: {podId: "%s"}) }' % pod['id'], key)
            print(f'terminated {pod["id"]} after {mins:.1f} min '
                  f'(~${pod["costPerHr"]*mins/60:.2f})')


if __name__ == '__main__':
    main()
