"""LLM problems in LeetCode format (PyTorch): fixed signature -> implement -> hidden tests vs torch reference.

Rules: only basic tensor ops (no torch.nn.functional for the thing being asked, no nn modules).
Each test compares against the official torch implementation / autograd / a naive loop reference, incl. edge cases.

Problems
  P01 softmax / log_softmax (stable, any dim)
  P02 cross_entropy (ignore_index, label_smoothing, reduction)
  P03 causal LM loss: shift, padding mask, response-only (SFT) mask, perplexity
  P04 cross-entropy backward by hand: dlogits = (softmax - onehot) / N
  P05 LayerNorm / RMSNorm
  P06 GELU / SiLU / SwiGLU FFN
  P07 dropout (train/eval, inverted scaling)
  P08 scaled dot-product attention (causal with T != S, bool mask, fully-masked rows)
  P09 RoPE (+ offset) / sinusoidal PE
  P10 MHA / GQA / MQA forward (n_kv_heads)
  P11 KV-cache decode loop == full forward
  P12 sampling: temperature / top-k / top-p
  P13 AdamW step
  P14 global-norm gradient clipping + warmup-cosine LR schedule
  P15 KL: exact from logits, and k1 / k3 sample estimators
  P16 sequence log-probs + DPO loss + Bradley-Terry reward-model loss
  P17 GRPO group advantages + PPO clipped token-level loss
  P18 MoE top-k router + Switch load-balancing loss
  P19 LoRA linear (forward, merge)
  P20 BPE tokenizer: train merges, encode, decode
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F


# P01 ------------------------------------------------------------------
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = x - x.amax(dim=dim, keepdim=True)
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = x - x.amax(dim=dim, keepdim=True)
    return z - z.exp().sum(dim=dim, keepdim=True).log()


# P02 ------------------------------------------------------------------
def cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
    """logits (N,C), target (N,) long. Matches F.cross_entropy."""
    logp = log_softmax(logits, -1)
    valid = target != ignore_index
    t = target.clamp(min=0)                                   # safe gather index for ignored rows
    nll = -logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(-1)                                   # CE against uniform
    loss = ((1 - label_smoothing) * nll + label_smoothing * smooth) * valid
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / valid.sum()                           # mean over non-ignored tokens only


# P03 ------------------------------------------------------------------
def causal_lm_loss(logits, input_ids, attention_mask, prompt_lens=None):
    """logits (B,T,V) for input_ids (B,T). Right padding (attention_mask 0 = pad).
    prompt_lens (B,): if given, only tokens at positions >= prompt_len count (SFT on responses).
    Returns (mean NLL over counted tokens, perplexity)."""
    B, T, V = logits.shape
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if prompt_lens is not None:
        pos = torch.arange(T).unsqueeze(0)
        labels[pos < prompt_lens.unsqueeze(1)] = -100
    loss = cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))
    return loss, loss.exp()


# P04 ------------------------------------------------------------------
def cross_entropy_backward(logits, target):
    """grad of mean CE w.r.t. logits (N,C)."""
    g = softmax(logits, -1)
    g[torch.arange(len(target)), target] -= 1
    return g / len(target)


# P05 ------------------------------------------------------------------
def layer_norm(x, weight, bias, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)             # biased variance!
    return (x - mu) / torch.sqrt(var + eps) * weight + bias


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


# P06 ------------------------------------------------------------------
def gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_ffn(x, w_gate, w_up, w_down):
    """x (..., D); w_gate/w_up (D, H); w_down (H, D). Llama FFN."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


# P07 ------------------------------------------------------------------
def dropout(x, p, training):
    if not training or p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)
    keep = (torch.rand(x.shape, dtype=x.dtype) >= p).to(x.dtype)
    return x * keep / (1 - p)


# P08 ------------------------------------------------------------------
def sdpa(q, k, v, attn_mask=None, is_causal=False):
    """q (...,T,d), k/v (...,S,d). attn_mask: bool (True = keep) broadcastable to (...,T,S).
    Causal with T != S: queries are the LAST T positions -> tril(diagonal=S-T). Fully-masked rows -> 0."""
    T, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    keep = torch.ones(T, S, dtype=torch.bool)
    if is_causal:
        keep = keep.tril(diagonal=S - T)
    if attn_mask is not None:
        keep = keep & attn_mask
    scores = scores.masked_fill(~keep, float("-inf"))
    w = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)  # softmax of all -inf = NaN -> 0
    return w @ v


# P09 ------------------------------------------------------------------
def apply_rope(x, offset=0, base=10000.0):
    """x (..., T, d): rotate pairs (2i, 2i+1) by angle pos * base^(-2i/d), pos = offset + t."""
    T, d = x.shape[-2], x.shape[-1]
    pos = torch.arange(offset, offset + T, dtype=x.dtype).unsqueeze(1)
    ang = pos * base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def sinusoidal_pe(T, d, base=10000.0):
    pos = torch.arange(T, dtype=torch.float64).unsqueeze(1)
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


# P10 ------------------------------------------------------------------
def gqa_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, causal=True):
    """x (B,T,D); wq (D,D), wk/wv (D, n_kv*Dh), wo (D,D). n_kv == n_heads -> MHA, 1 -> MQA."""
    B, T, D = x.shape
    Dh = D // n_heads
    G = n_heads // n_kv_heads
    q = (x @ wq).view(B, T, n_heads, Dh).transpose(1, 2)              # (B,H,T,Dh)
    k = (x @ wk).view(B, T, n_kv_heads, Dh).transpose(1, 2)           # (B,Hkv,T,Dh)
    v = (x @ wv).view(B, T, n_kv_heads, Dh).transpose(1, 2)
    k = k.repeat_interleave(G, dim=1)                                 # head h uses kv head h // G
    v = v.repeat_interleave(G, dim=1)
    o = sdpa(q, k, v, is_causal=causal)
    return o.transpose(1, 2).reshape(B, T, D) @ wo


# P11 ------------------------------------------------------------------
def decode_with_kv_cache(x, wq, wk, wv, n_heads):
    """Single-layer causal attention computed token by token with a KV cache. x (B,T,D) -> (B,T,D).
    Must equal the full causal forward. RoPE applied with absolute positions before caching k."""
    B, T, D = x.shape
    Dh = D // n_heads
    k_cache = torch.zeros(B, n_heads, 0, Dh, dtype=x.dtype)
    v_cache = torch.zeros(B, n_heads, 0, Dh, dtype=x.dtype)
    outs = []
    for t in range(T):
        xt = x[:, t:t + 1]
        q = apply_rope((xt @ wq).view(B, 1, n_heads, Dh).transpose(1, 2), offset=t)
        k = apply_rope((xt @ wk).view(B, 1, n_heads, Dh).transpose(1, 2), offset=t)
        v = (xt @ wv).view(B, 1, n_heads, Dh).transpose(1, 2)
        k_cache = torch.cat([k_cache, k], 2)
        v_cache = torch.cat([v_cache, v], 2)
        outs.append(sdpa(q, k_cache, v_cache))                         # new query sees all cached keys
    return torch.cat(outs, 2).transpose(1, 2).reshape(B, T, D)


# P12 ------------------------------------------------------------------
def filter_logits(logits, temperature=1.0, k=0, p=1.0):
    """logits (B,V) -> filtered logits (removed = -inf). temperature 0 -> keep only argmax.
    top-p keeps the smallest prefix with cumulative prob >= p (always >= 1 token)."""
    if temperature == 0:
        out = torch.full_like(logits, float("-inf"))
        return out.scatter(-1, logits.argmax(-1, keepdim=True), 0.0)
    out = logits / temperature
    if k > 0:
        kth = out.topk(k, -1).values[..., -1:]
        out = out.masked_fill(out < kth, float("-inf"))
    if p < 1.0:
        sl, idx = out.sort(-1, descending=True)
        probs = sl.softmax(-1)
        remove = probs.cumsum(-1) - probs >= p                # mass BEFORE this token already >= p
        remove[..., 0] = False
        out = out.scatter(-1, idx, sl.masked_fill(remove, float("-inf")))
    return out


# P13 ------------------------------------------------------------------
def adamw_step(p, g, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
    """In place on p, m, v. t is the 1-based step. Decoupled weight decay (not added to g)."""
    b1, b2 = betas
    p.mul_(1 - lr * wd)
    m.mul_(b1).add_((1 - b1) * g)
    v.mul_(b2).add_((1 - b2) * g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p.sub_(lr * mhat / (vhat.sqrt() + eps))


# P14 ------------------------------------------------------------------
def clip_grad_norm(grads, max_norm, eps=1e-6):
    """grads: list of tensors, modified in place. Returns the total norm BEFORE clipping."""
    total = torch.sqrt(sum((g.double() ** 2).sum() for g in grads))
    scale = min(1.0, max_norm / (total.item() + eps))
    for g in grads:
        g.mul_(scale)
    return total


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# P15 ------------------------------------------------------------------
def kl_from_logits(p_logits, q_logits):
    """exact KL(P || Q) per row, mean over rows."""
    lp, lq = log_softmax(p_logits, -1), log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def kl_estimators(logp, logq):
    """samples x ~ P with log p(x), log q(x). Returns (k1, k3) per-sample estimates of KL(P||Q).
    k1 = log p - log q (unbiased, can be negative); k3 = r - 1 - log r with r = q/p (unbiased, >= 0, low variance)."""
    log_r = logq - logp
    return -log_r, log_r.exp() - 1 - log_r


# P16 ------------------------------------------------------------------
def sequence_logprob(logits, input_ids, loss_mask):
    """sum of log p(token) over positions where loss_mask[:,1:] == 1. logits (B,T,V) -> (B,)"""
    logp = log_softmax(logits[:, :-1], -1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (logp * loss_mask[:, 1:]).sum(-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """inputs: (B,) sequence log-probs. loss = -log σ(β[(π_c - ref_c) - (π_r - ref_r)]), stable via logsigmoid form."""
    z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return (torch.clamp(-z, min=0) + torch.log1p(torch.exp(-z.abs()))).mean()      # = softplus(-z)


def bradley_terry_loss(r_chosen, r_rejected):
    z = r_chosen - r_rejected
    return (torch.clamp(-z, min=0) + torch.log1p(torch.exp(-z.abs()))).mean()


# P17 ------------------------------------------------------------------
def grpo_advantages(rewards, group_size, eps=1e-6):
    """rewards (B*G,) laid out group by group -> advantage per sample = (r - mean_group) / (std_group + eps)."""
    r = rewards.view(-1, group_size)
    a = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + eps)
    return a.view(-1)


def ppo_clip_loss(logp_new, logp_old, advantages, mask, clip=0.2):
    """token-level: logp_* (B,T), advantages (B,) broadcast to tokens or (B,T), mask (B,T) response tokens.
    loss = -mean over masked tokens of min(r A, clip(r) A)."""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    ratio = (logp_new - logp_old).exp()
    obj = torch.min(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -(obj * mask).sum() / mask.sum()


# P18 ------------------------------------------------------------------
def moe_forward(x, router_w, experts, k=2):
    """x (N,D), router_w (D,E), experts: list of callables D->D. Softmax over all E, take top-k, renormalise.
    Returns (y, aux_loss) with Switch aux = E * sum_i f_i * P_i
    (f_i = fraction of top-k assignments to expert i, P_i = mean router prob of expert i)."""
    N, E = x.shape[0], router_w.shape[1]
    probs = softmax(x @ router_w, -1)                          # (N,E)
    topv, topi = probs.topk(k, -1)                             # (N,k)
    gates = topv / topv.sum(-1, keepdim=True)
    y = torch.zeros_like(x)
    for e in range(E):                                         # loop experts, batch tokens routed to each
        tok, slot = (topi == e).nonzero(as_tuple=True)
        if len(tok):
            y.index_add_(0, tok, gates[tok, slot].unsqueeze(1) * experts[e](x[tok]))
    f = torch.bincount(topi.reshape(-1), minlength=E).to(x.dtype) / (N * k)
    P = probs.mean(0)
    return y, E * (f * P).sum()


# P19 ------------------------------------------------------------------
def lora_linear(x, W, A, B, alpha, r):
    """y = x W^T + (alpha/r) x A^T B^T. W (out,in) frozen, A (r,in), B (out,r) (B init to zeros)."""
    return x @ W.T + (alpha / r) * (x @ A.T) @ B.T


def lora_merge(W, A, B, alpha, r):
    return W + (alpha / r) * B @ A


# P20 ------------------------------------------------------------------
def bpe_train(text, n_merges):
    """byte-level BPE. Returns list of merges [((a, b), new_id), ...], new ids start at 256.
    Ties broken by first occurrence (Counter.most_common keeps insertion order)."""
    ids = list(text.encode())
    merges = []
    for i in range(n_merges):
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        pair = pairs.most_common(1)[0][0]
        new = 256 + i
        merges.append((pair, new))
        ids = _merge(ids, pair, new)
    return merges


def _merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def bpe_encode(text, merges):
    ids = list(text.encode())
    for pair, new in merges:                                   # apply in training order
        ids = _merge(ids, pair, new)
    return ids


def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new in merges:
        vocab[new] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# ======================================================================= tests
def close(a, b, tol=1e-6):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    assert torch.allclose(a, b.to(a.dtype), atol=tol, rtol=tol), (a - b).abs().max()


def test_all():
    torch.manual_seed(0)
    dt = torch.double

    x = torch.randn(4, 7, 10, dtype=dt) * 5
    x[0, 0, 0] = 1000.0                                        # overflow trap
    for d in (-1, 1, 0):
        close(softmax(x, d), torch.softmax(x, d)); close(log_softmax(x, d), torch.log_softmax(x, d))
    print("P01 softmax/log_softmax ok")

    logits = torch.randn(16, 9, dtype=dt) * 3
    tgt = torch.randint(0, 9, (16,)); tgt[[2, 5]] = -100
    for ls in (0.0, 0.1):
        for red in ("mean", "sum", "none"):
            close(cross_entropy(logits, tgt, label_smoothing=ls, reduction=red),
                  F.cross_entropy(logits, tgt, label_smoothing=ls, reduction=red))
    print("P02 cross_entropy ok")

    B, T, V = 3, 10, 20
    lg = torch.randn(B, T, V, dtype=dt); ids = torch.randint(0, V, (B, T))
    am = torch.ones(B, T, dtype=torch.long); am[1, 7:] = 0; am[2, 4:] = 0
    pl = torch.tensor([3, 2, 1])
    lab = ids.clone(); lab[am == 0] = -100; lab[torch.arange(T) < pl[:, None]] = -100
    ref = F.cross_entropy(lg[:, :-1].reshape(-1, V), lab[:, 1:].reshape(-1), ignore_index=-100)
    loss, ppl = causal_lm_loss(lg, ids, am, pl)
    close(loss, ref); close(ppl, ref.exp())
    print("P03 causal LM loss ok (shift + pad mask + prompt mask)")

    lgr = torch.randn(8, 5, dtype=dt, requires_grad=True); t8 = torch.randint(0, 5, (8,))
    F.cross_entropy(lgr, t8).backward()
    close(cross_entropy_backward(lgr.detach(), t8), lgr.grad)
    print("P04 CE backward ok")

    w, bb = torch.randn(10, dtype=dt), torch.randn(10, dtype=dt)
    close(layer_norm(x, w, bb), F.layer_norm(x, (10,), w, bb))
    close(rms_norm(x, w), F.rms_norm(x, (10,), w, eps=1e-6))
    print("P05 LayerNorm/RMSNorm ok")

    close(gelu(x), F.gelu(x)); close(gelu(x, "tanh"), F.gelu(x, approximate="tanh")); close(silu(x), F.silu(x))
    wg, wu, wd_ = torch.randn(10, 16, dtype=dt), torch.randn(10, 16, dtype=dt), torch.randn(16, 10, dtype=dt)
    close(swiglu_ffn(x, wg, wu, wd_), F.linear(F.silu(F.linear(x, wg.T)) * F.linear(x, wu.T), wd_.T))
    print("P06 GELU/SiLU/SwiGLU ok")

    big = torch.ones(200000, dtype=dt)
    d = dropout(big, 0.3, True)
    assert abs((d == 0).double().mean().item() - 0.3) < 0.01 and abs(d.mean().item() - 1) < 0.01
    assert torch.equal(dropout(big, 0.3, False), big)
    print("P07 dropout ok")

    q, k, v = torch.randn(2, 3, 5, 8, dtype=dt), torch.randn(2, 3, 7, 8, dtype=dt), torch.randn(2, 3, 7, 8, dtype=dt)
    close(sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v))
    close(sdpa(q, k, v, is_causal=True),
          F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(5, 7, dtype=torch.bool).tril(2)))
    m = torch.rand(5, 7) > 0.3; m[:, 0] = True
    close(sdpa(q, k, v, attn_mask=m), F.scaled_dot_product_attention(q, k, v, attn_mask=m))
    m[1] = False
    assert sdpa(q, k, v, attn_mask=m)[..., 1, :].abs().max() == 0
    print("P08 sdpa ok (causal T!=S, bool mask, fully masked row -> 0)")

    xr = torch.randn(2, 6, 8, dtype=dt)
    close(apply_rope(xr).norm(dim=-1), xr.norm(dim=-1))
    qq, kk = torch.randn(1, 1, 8, dtype=dt), torch.randn(1, 1, 8, dtype=dt)
    close((apply_rope(qq, 3) * apply_rope(kk, 1)).sum(), (apply_rope(qq, 10) * apply_rope(kk, 8)).sum())
    close(apply_rope(xr, 0)[:, 2:], apply_rope(xr[:, 2:], 2))
    assert torch.allclose(sinusoidal_pe(6, 8)[0], torch.tensor([0., 1.] * 4, dtype=dt))
    print("P09 RoPE ok (norm kept, relative position, offset) / sinusoidal ok")

    D, H = 32, 8
    xa = torch.randn(2, 6, D, dtype=dt)
    for Hkv in (8, 2, 1):
        Dh = D // H
        wq, wk, wv, wo = (torch.randn(D, D, dtype=dt), torch.randn(D, Hkv * Dh, dtype=dt),
                          torch.randn(D, Hkv * Dh, dtype=dt), torch.randn(D, D, dtype=dt))
        qq_ = (xa @ wq).view(2, 6, H, Dh).transpose(1, 2)
        kk_ = (xa @ wk).view(2, 6, Hkv, Dh).transpose(1, 2)
        vv_ = (xa @ wv).view(2, 6, Hkv, Dh).transpose(1, 2)
        ref = F.scaled_dot_product_attention(qq_, kk_, vv_, is_causal=True, enable_gqa=True)
        close(gqa_attention(xa, wq, wk, wv, wo, H, Hkv), ref.transpose(1, 2).reshape(2, 6, D) @ wo)
    print("P10 MHA/GQA/MQA ok (vs SDPA enable_gqa)")

    wq, wk, wv = (torch.randn(D, D, dtype=dt) for _ in range(3))
    Bq, Tq = 2, 9
    xs = torch.randn(Bq, Tq, D, dtype=dt)
    qf = apply_rope((xs @ wq).view(Bq, Tq, H, -1).transpose(1, 2))
    kf = apply_rope((xs @ wk).view(Bq, Tq, H, -1).transpose(1, 2))
    vf = (xs @ wv).view(Bq, Tq, H, -1).transpose(1, 2)
    full = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True).transpose(1, 2).reshape(Bq, Tq, D)
    close(decode_with_kv_cache(xs, wq, wk, wv, H), full)
    print("P11 KV-cache decode == full causal forward ok")

    lg4 = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=dt))
    keep = lambda o: torch.isfinite(o)[0].tolist()
    assert keep(filter_logits(lg4, k=2)) == [True, True, False, False]
    assert keep(filter_logits(lg4, p=0.75)) == [True, True, False, False]
    assert keep(filter_logits(lg4, p=0.85)) == [True, True, True, False]
    assert keep(filter_logits(lg4, p=0.1)) == [True, False, False, False]
    assert keep(filter_logits(lg4, temperature=0)) == [True, False, False, False]
    close(filter_logits(lg4, temperature=2.0).softmax(-1), (lg4 / 2).softmax(-1))
    print("P12 temperature/top-k/top-p ok (boundaries, T=0)")

    p_ref = torch.randn(5, dtype=dt, requires_grad=True); p_me = p_ref.detach().clone()
    opt = torch.optim.AdamW([p_ref], lr=0.1, weight_decay=0.1)
    mm, vv = torch.zeros(5, dtype=dt), torch.zeros(5, dtype=dt)
    for t in range(1, 6):
        g = torch.randn(5, dtype=dt)
        p_ref.grad = g.clone(); opt.step()
        adamw_step(p_me, g, mm, vv, t, lr=0.1, wd=0.1)
    close(p_me, p_ref.detach())
    print("P13 AdamW ok (5 steps vs torch.optim.AdamW)")

    ps = [torch.randn(3, 4, dtype=dt, requires_grad=True), torch.randn(7, dtype=dt, requires_grad=True)]
    gs = [torch.randn_like(p) * 10 for p in ps]
    for p, g in zip(ps, gs):
        p.grad = g.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    my_norm = clip_grad_norm(gs, 1.0)
    close(my_norm, ref_norm); close(gs[0], ps[0].grad); close(gs[1], ps[1].grad)
    assert abs(lr_at(0, 1.0, 10, 100) - 0.1) < 1e-12 and abs(lr_at(9, 1.0, 10, 100) - 1.0) < 1e-12
    assert abs(lr_at(10, 1.0, 10, 100) - 1.0) < 1e-12 and abs(lr_at(100, 1.0, 10, 100, 0.1) - 0.1) < 1e-12
    assert abs(lr_at(55, 1.0, 10, 100) - 0.5) < 1e-12
    print("P14 clip_grad_norm (vs torch) + warmup-cosine ok")

    a, b = torch.randn(5, 8, dtype=dt), torch.randn(5, 8, dtype=dt)
    close(kl_from_logits(a, b), F.kl_div(torch.log_softmax(b, -1), torch.log_softmax(a, -1), log_target=True, reduction="batchmean"))
    pl1, ql1 = torch.log_softmax(a[0], -1), torch.log_softmax(b[0], -1)
    xs_ = torch.multinomial(pl1.exp(), 400000, replacement=True, generator=torch.Generator().manual_seed(1))
    k1, k3 = kl_estimators(pl1[xs_], ql1[xs_])
    true = (pl1.exp() * (pl1 - ql1)).sum().item()
    assert abs(k1.mean().item() - true) < 0.02 and abs(k3.mean().item() - true) < 0.02 and (k3 >= 0).all()
    print(f"P15 KL ok (exact vs F.kl_div; MC k1 {k1.mean():.3f}, k3 {k3.mean():.3f}, true {true:.3f})")

    lgp = torch.randn(B, T, V, dtype=dt)
    lmask = am.clone().double(); lmask[torch.arange(T) < pl[:, None]] = 0
    lab2 = ids.clone(); lab2[lmask == 0] = -100
    ref_lp = -F.cross_entropy(lgp[:, :-1].reshape(-1, V), lab2[:, 1:].reshape(-1), reduction="none").view(B, -1).sum(-1)
    close(sequence_logprob(lgp, ids, lmask), ref_lp)
    pc, pr, rc, rr = (torch.randn(4, dtype=dt) * 20 for _ in range(4))
    close(dpo_loss(pc, pr, rc, rr, 0.1), -F.logsigmoid(0.1 * ((pc - rc) - (pr - rr))).mean())
    close(dpo_loss(rc, rr, rc, rr), torch.tensor(math.log(2)))                    # policy == ref -> log 2
    big_r = torch.tensor([800.0], dtype=dt)
    assert torch.isfinite(bradley_terry_loss(-big_r, big_r))
    close(bradley_terry_loss(pc, pr), -F.logsigmoid(pc - pr).mean())
    print("P16 sequence logprob + DPO + Bradley-Terry ok")

    rw = torch.tensor([1., 0., 0., 1., 5., 5., 5., 5.], dtype=dt)
    adv = grpo_advantages(rw, 4)
    close(adv[:4].mean(), torch.tensor(0.0)); close(adv[:4].std(), torch.tensor(1.0), 1e-5)
    assert adv[4:].abs().max() == 0                                                # all-equal group -> zero signal
    lo, lnew = torch.randn(2, 6, dtype=dt), torch.randn(2, 6, dtype=dt)
    mk = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]], dtype=dt)
    A = torch.tensor([1.0, -2.0], dtype=dt)
    close(ppo_clip_loss(lo, lo, A, mk), -(A[:, None] * mk).sum() / mk.sum())       # ratio 1 -> -mean(A)
    lnew_ = (lo + torch.tensor([[1.0] * 6, [0.0] * 6], dtype=dt)).requires_grad_()  # ratio e for seq 0 (A>0)
    ppo_clip_loss(lnew_, lo, A, mk).backward()
    assert lnew_.grad[0].abs().max() == 0                                          # clipped -> no gradient
    print("P17 GRPO advantages + PPO clip ok")

    N, Dm, E = 12, 6, 4
    xm = torch.randn(N, Dm, dtype=dt); rw_ = torch.randn(Dm, E, dtype=dt)
    Ws = [torch.randn(Dm, Dm, dtype=dt) for _ in range(E)]
    experts = [lambda z, W=W: torch.tanh(z @ W) for W in Ws]
    y, aux = moe_forward(xm, rw_, experts, 2)
    pr_ = torch.softmax(xm @ rw_, -1)
    for n in range(N):                                                             # naive per-token reference
        tv, ti = pr_[n].topk(2)
        ref = sum((tv[j] / tv.sum()) * experts[ti[j]](xm[n:n + 1])[0] for j in range(2))
        close(y[n], ref)
    _, aux_u = moe_forward(xm, torch.zeros(Dm, E, dtype=dt), experts, 2)
    close(aux_u, torch.tensor(1.0))                                                # uniform router -> aux = 1
    print(f"P18 MoE top-2 ok (vs per-token loop), aux {aux.item():.3f}, uniform aux = 1")

    W = torch.randn(5, 8, dtype=dt); A_ = torch.randn(2, 8, dtype=dt); B0 = torch.zeros(5, 2, dtype=dt)
    xl = torch.randn(3, 8, dtype=dt)
    close(lora_linear(xl, W, A_, B0, 16, 2), xl @ W.T)                            # B = 0 -> identical to base
    B1 = torch.randn(5, 2, dtype=dt)
    close(lora_linear(xl, W, A_, B1, 16, 2), xl @ lora_merge(W, A_, B1, 16, 2).T)
    print("P19 LoRA ok (zero-init identity, merge == unmerged)")

    text = "low lower lowest newer newest wider widest " * 5
    merges = bpe_train(text, 20)
    enc = bpe_encode(text, merges)
    assert bpe_decode(enc, merges) == text and len(enc) < len(text.encode())
    assert bpe_decode(bpe_encode("lowest 你好", merges), merges) == "lowest 你好"     # unseen / multibyte round trip
    assert merges[0][0] in {(ord("w"), ord("e")), (ord("e"), ord("s")), (ord("l"), ord("o")), (ord("o"), ord("w"))}
    print(f"P20 BPE ok ({len(text.encode())} bytes -> {len(enc)} tokens, round trip)")
    print("OK")


if __name__ == "__main__":
    test_all()
