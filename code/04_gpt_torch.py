"""Minimal GPT in PyTorch: RoPE, RMSNorm, causal MHA with KV cache, SwiGLU MLP, AdamW + warmup/cosine,
grad clipping, grad accumulation, and sampling (temperature / top-k / top-p).

Trains a char-level model on a tiny repeated text and checks:
  1) loss goes down
  2) KV-cache generation == full-recompute generation (greedy)

Interview checklist:
- pre-norm residual: x = x + attn(norm(x)); x = x + mlp(norm(x))
- weight tying between token embedding and lm_head
- AdamW: decoupled weight decay; common choice: decay only >=2D weights (matmuls, embeddings), not norms/biases
- KV cache: per layer store k,v of shape (B, H, T_past, Dh); new token attends to all of them; no mask needed for 1 query
- per-token cost with cache: O(T * D) attention instead of O(T^2 * D)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(T, dh, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2).float() / dh)       # (dh/2,)
    ang = torch.outer(torch.arange(T).float(), inv)                  # (T, dh/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (B, H, T, dh); rotate pairs (x0,x1),(x2,x3),..."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # each (B,H,T,dh)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        # causal mask that also works when T < Tk (decoding with cache)
        mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), Attention(d, h)
        self.n2, self.mlp = RMSNorm(d), SwiGLU(d, 4 * d)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=64, h=4, L=2, max_T=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight                     # weight tying
        cos, sin = rope_cache(max_T, d // h)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, caches=None, pos=0):
        T = idx.shape[1]
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i])
        return self.head(self.norm(x))


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """logits (B, V) -> (B, 1)."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(-1, descending=True)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p                     # keep the smallest prefix with mass >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def generate_naive(model, idx, n, **kw):
    for _ in range(n):
        logits = model(idx)[:, -1]
        idx = torch.cat([idx, sample_next(logits, **kw)], 1)
    return idx


@torch.no_grad()
def generate_kv(model, idx, n, **kw):
    caches = [{} for _ in model.blocks]
    logits = model(idx, caches, pos=0)[:, -1]            # prefill
    out = idx
    for _ in range(n):
        nxt = sample_next(logits, **kw)
        out = torch.cat([out, nxt], 1)
        logits = model(nxt, caches, pos=out.shape[1] - 1)[:, -1]   # decode one token
    return out


def lr_at(step, max_lr, warmup, total, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def configure_adamw(model, lr, wd=0.1):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


if __name__ == "__main__":
    text = "the quick brown fox jumps over the lazy dog. " * 50
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    model = GPT(len(chars))
    opt = configure_adamw(model, 3e-3)

    T, B, steps, accum = 32, 16, 300, 2
    first = None
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, 3e-3, 30, steps)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):                                   # gradient accumulation
            ix = torch.randint(0, len(data) - T - 1, (B,))
            x = torch.stack([data[i:i + T] for i in ix])
            y = torch.stack([data[i + 1:i + T + 1] for i in ix])
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten()) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item() * accum
        if step % 100 == 0:
            print(f"step {step} loss {loss.item() * accum:.3f}")
    final = loss.item() * accum
    print(f"loss {first:.3f} -> {final:.3f}")
    assert final < first * 0.3

    model.eval()
    prompt = torch.tensor([[stoi[c] for c in "the quick "]])
    a = generate_naive(model, prompt, 30, temperature=0)
    b = generate_kv(model, prompt, 30, temperature=0)
    decode = lambda t: "".join(chars[i] for i in t[0].tolist())
    print("naive:", repr(decode(a)))
    print("kv   :", repr(decode(b)))
    assert torch.equal(a, b)
    torch.manual_seed(1)
    print("top-p:", repr(decode(generate_kv(model, prompt, 30, temperature=0.8, top_k=10, top_p=0.9))))
    print("OK")
