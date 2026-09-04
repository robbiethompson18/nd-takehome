"""Decoder-only transformer over the ndtok vocabulary.

The positional scheme is a flag because it is the Stage-2 variable.  Stage 1 trains on proofs of
<= 6 lines (<= ~141 tokens); Stage 2 has to emit 12-16 line proofs (~300 tokens).  So:

  nope : no positional signal at all.  A causal decoder recovers order from the mask alone, and
         no weight is indexed by absolute position, so extending the context costs nothing.
  rope : rotary (relative) embeddings, the conventional baseline to compare against.

Learned absolute position embeddings are deliberately absent: their rows past the longest training
sequence never receive a gradient, so at Stage 2 they would feed noise into every attention score.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ndtok import PAD, QED, VOCAB


@dataclass
class Config:
    vocab: int = len(VOCAB)
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    max_len: int = 512
    pos: str = 'nope'            # 'nope' | 'rope'
    rope_base: float = 10000.0

    @property
    def head_dim(self):
        return self.d_model // self.n_head


def rope_tables(cfg, device, dtype=torch.float32):
    """cos/sin of shape [max_len, head_dim/2], half-split (LLaMA) convention."""
    hd = cfg.head_dim
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    ang = torch.outer(torch.arange(cfg.max_len, device=device, dtype=dtype), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, hd]."""
    T = x.shape[2]
    c, s = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).type_as(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)
        if cfg.pos == 'rope':
            cos, sin = rope_tables(cfg, 'cpu')
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, key_pad=None):
        """idx: [B, T] token ids.  key_pad: [B, T] bool, True where the token is real.

        Training right-pads, so causal masking already hides the padding and key_pad is None.
        Batched generation left-pads, so it must pass key_pad."""
        T = idx.shape[1]
        assert T <= self.cfg.max_len, f'sequence {T} > max_len {self.cfg.max_len}'
        mask = None
        if key_pad is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
            eye = torch.eye(T, dtype=torch.bool, device=idx.device)
            # `| eye` keeps all-masked rows (pad queries) from producing NaNs; it adds nothing
            # for real queries, which already attend to themselves.
            mask = (causal & key_pad[:, None, None, :]) | eye[None, None]
        cos = sin = None
        if self.cfg.pos == 'rope':
            cos, sin = self.cos, self.sin
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompts, max_new=320, temperature=0.0, generator=None):
        """prompts: list of id lists.  Returns one generated id list per prompt, up to and
        including the first QED.  No KV cache: every step re-runs the whole prefix.  That costs
        B*T^2*N flops (~300 TFLOP for 1024 rollouts of 300 tokens) which a GPU eats in seconds."""
        self.eval()
        dev = next(self.parameters()).device
        B, P = len(prompts), max(len(p) for p in prompts)
        idx = torch.full((B, P), PAD, dtype=torch.long, device=dev)
        key_pad = torch.zeros((B, P), dtype=torch.bool, device=dev)
        for i, p in enumerate(prompts):           # left-pad so every row generates at the edge
            idx[i, P - len(p):] = torch.tensor(p, device=dev)
            key_pad[i, P - len(p):] = True
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        out = []
        for _ in range(min(max_new, self.cfg.max_len - P)):
            logits = self(idx, key_pad)[:, -1].float()
            if temperature > 0:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)
                nxt = nxt.squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, QED), nxt)   # finished rows idle on QED
            done |= nxt == QED
            out.append(nxt)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
            key_pad = torch.cat([key_pad, torch.ones((B, 1), dtype=torch.bool, device=dev)], 1)
            if done.all():
                break
        gen = torch.stack(out, dim=1).tolist() if out else [[] for _ in range(B)]
        return [g[:g.index(QED) + 1] if QED in g else g for g in gen]


def save(path, model, extra=None):
    torch.save({'cfg': asdict(model.cfg), 'state': model.state_dict(), 'extra': extra or {}}, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Model(Config(**ck['cfg'])).to(device)
    model.load_state_dict(ck['state'])
    return model, ck['extra']
