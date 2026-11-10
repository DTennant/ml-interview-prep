"""ML problems in LeetCode format (PyTorch): fixed signature -> implement -> hidden tests vs torch reference.

Rules for every problem: use only basic tensor ops (no torch.nn.functional for the thing being asked, no nn modules).
Each test compares against the official torch implementation, including edge cases (masks, ignore_index, extreme
logits, odd shapes).

Problems
  P01 softmax / log_softmax (stable, any dim)
  P02 cross_entropy (class indices, ignore_index, label_smoothing, reduction)
  P03 binary_cross_entropy_with_logits (stable, pos_weight)
  P04 kl_div between two logit tensors
  P05 layer_norm / rms_norm
  P06 batch_norm forward (train: batch stats + running stats update; eval: running stats)
  P07 dropout (train/eval, inverted scaling)
  P08 gelu (exact + tanh), silu, swiglu
  P09 scaled_dot_product_attention (causal, boolean mask, -inf rows)
  P10 sinusoidal positional encoding / RoPE
  P11 conv2d (stride, padding) via unfold-free loops-over-kernel
  P12 max_pool2d / avg_pool2d
  P13 linear layer manual backward (dX, dW, db) vs autograd
  P14 AdamW single step vs torch.optim.AdamW
  P15 top-k / top-p (nucleus) filtering for sampling
  P16 InfoNCE / CLIP contrastive loss
  P17 IoU matrix + NMS (vs torchvision-style reference written plainly)
  P18 accuracy@k, confusion matrix, ROC-AUC (rank-based)
  P19 embedding_bag (sum/mean by offsets)
  P20 cosine-similarity top-k retrieval
"""
import math

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
    C = logits.shape[-1]
    logp = log_softmax(logits, -1)
    valid = target != ignore_index
    t = target.clamp(min=0)                                   # safe gather index for ignored rows
    nll = -logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(-1)                                   # CE against uniform
    loss = (1 - label_smoothing) * nll + label_smoothing * smooth
    loss = loss * valid
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / valid.sum()                           # mean over non-ignored (nan if all ignored, like torch)


# P03 ------------------------------------------------------------------
def bce_with_logits(x, y, pos_weight=None):
    """mean over elements. stable: log(1+e^-|x|) + max(x,0) - x*y form, generalised for pos_weight."""
    log_sig = -(torch.clamp(-x, min=0) + torch.log1p(torch.exp(-x.abs())))   # log σ(x) = -softplus(-x), stable
    log_one_minus = log_sig - x                                                                              # log(1-σ(x))
    pw = 1.0 if pos_weight is None else pos_weight
    return -(pw * y * log_sig + (1 - y) * log_one_minus).mean()


# P04 ------------------------------------------------------------------
def kl_div_logits(p_logits, q_logits):
    """KL(P || Q) per row, averaged over rows ('batchmean')."""
    lp, lq = log_softmax(p_logits, -1), log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


# P05 ------------------------------------------------------------------
def layer_norm(x, weight, bias, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)             # biased variance!
    return (x - mu) / torch.sqrt(var + eps) * weight + bias


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


# P06 ------------------------------------------------------------------
def batch_norm(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5):
    """x (N,C) or (N,C,H,W). Updates running stats in place when training (unbiased var for running stats)."""
    dims = [0] + list(range(2, x.dim()))
    shape = [1, -1] + [1] * (x.dim() - 2)
    if training:
        mean = x.mean(dims)
        var = x.var(dims, unbiased=False)
        n = x.numel() // x.shape[1]
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * mean)
            running_var.mul_(1 - momentum).add_(momentum * var * n / (n - 1))
    else:
        mean, var = running_mean, running_var
    return (x - mean.view(shape)) / torch.sqrt(var.view(shape) + eps) * weight.view(shape) + bias.view(shape)


# P07 ------------------------------------------------------------------
def dropout(x, p, training, generator=None):
    if not training or p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)
    keep = (torch.rand(x.shape, generator=generator, dtype=x.dtype) >= p).to(x.dtype)
    return x * keep / (1 - p)


# P08 ------------------------------------------------------------------
def gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def silu(x):
    return x * torch.sigmoid(x)


def swiglu(x, w_gate, w_up, w_down):
    """x (..., D); w_gate/w_up (D, H); w_down (H, D)."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


# P09 ------------------------------------------------------------------
def sdpa(q, k, v, attn_mask=None, is_causal=False):
    """q (...,T,d), k/v (...,S,d). attn_mask: bool (True = keep) broadcastable to (...,T,S). Fully-masked rows -> 0."""
    T, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    keep = torch.ones(T, S, dtype=torch.bool)
    if is_causal:
        keep = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
    if attn_mask is not None:
        keep = keep & attn_mask
    scores = scores.masked_fill(~keep, float("-inf"))
    w = torch.softmax(scores, -1)
    w = torch.nan_to_num(w, nan=0.0)                          # rows with no valid key
    return w @ v


# P10 ------------------------------------------------------------------
def sinusoidal_pe(T, d, base=10000.0):
    pos = torch.arange(T, dtype=torch.float64).unsqueeze(1)
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


def apply_rope(x, offset=0, base=10000.0):
    """x (..., T, d), rotate pairs (2i, 2i+1) by angle pos * base^(-2i/d)."""
    T, d = x.shape[-2], x.shape[-1]
    pos = torch.arange(offset, offset + T, dtype=x.dtype).unsqueeze(1)
    ang = pos * base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# P11 ------------------------------------------------------------------
def conv2d(x, w, b=None, stride=1, padding=0):
    """x (N,Cin,H,W), w (Cout,Cin,kh,kw). Loop over kernel offsets (kh*kw iterations), everything else vectorised."""
    N, Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    x = F.pad(x, (padding, padding, padding, padding))
    Ho = (H + 2 * padding - kh) // stride + 1
    Wo = (W + 2 * padding - kw) // stride + 1
    out = torch.zeros(N, Cout, Ho, Wo, dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]       # (N,Cin,Ho,Wo)
            out += torch.einsum("nchw,oc->nohw", patch, w[:, :, i, j])
    if b is not None:
        out += b.view(1, -1, 1, 1)
    return out


# P12 ------------------------------------------------------------------
def pool2d(x, k, stride=None, mode="max"):
    stride = stride or k
    N, C, H, W = x.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    windows = [x[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] for i in range(k) for j in range(k)]
    st = torch.stack(windows, 0)
    return st.amax(0) if mode == "max" else st.mean(0)


# P13 ------------------------------------------------------------------
def linear_backward(x, w, dy):
    """y = x @ w.T + b, x (N,in), w (out,in), dy (N,out). Return dx, dw, db."""
    return dy @ w, dy.T @ x, dy.sum(0)


# P14 ------------------------------------------------------------------
def adamw_step(p, g, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
    """in-place on p, m, v. t is the 1-based step count. Decoupled weight decay applied before the Adam update."""
    b1, b2 = betas
    p.mul_(1 - lr * wd)
    m.mul_(b1).add_((1 - b1) * g)
    v.mul_(b2).add_((1 - b2) * g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p.sub_(lr * mhat / (vhat.sqrt() + eps))


# P15 ------------------------------------------------------------------
def top_k_top_p_filter(logits, k=0, p=1.0):
    """logits (B,V) -> logits with filtered entries set to -inf. Keep the smallest prefix with cumprob >= p
    (always keep at least the top-1)."""
    out = logits.clone()
    if k > 0:
        kth = out.topk(k, -1).values[..., -1:]
        out[out < kth] = float("-inf")
    if p < 1.0:
        sl, idx = out.sort(-1, descending=True)
        cum = sl.softmax(-1).cumsum(-1)
        remove = cum - sl.softmax(-1) >= p                    # prob mass BEFORE this token already >= p
        remove[..., 0] = False
        out.scatter_(-1, idx, sl.masked_fill(remove, float("-inf")))
    return out


# P16 ------------------------------------------------------------------
def clip_loss(img, txt, temperature=0.07):
    """img/txt (N,d) unnormalised. Symmetric InfoNCE; positives on the diagonal."""
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    logits = img @ txt.T / temperature
    labels = torch.arange(len(img))
    return (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2


# P17 ------------------------------------------------------------------
def box_iou(a, b):
    """a (N,4), b (M,4) xyxy -> (N,M)."""
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def nms(boxes, scores, iou_thr):
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool)
    iou = box_iou(boxes, boxes)
    for i in order.tolist():
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_thr
    return torch.tensor(keep, dtype=torch.long)


# P18 ------------------------------------------------------------------
def topk_accuracy(logits, target, k):
    return (logits.topk(k, -1).indices == target[:, None]).any(-1).float().mean()


def confusion_matrix(pred, target, C):
    return torch.bincount(target * C + pred, minlength=C * C).view(C, C)    # rows = true, cols = pred


def roc_auc(scores, labels):
    """P(score_pos > score_neg) with ties counted 1/2 -> Mann-Whitney U with average ranks."""
    s = scores.double()
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.double)
    for val in s.unique():                                     # average ranks for ties
        m = s == val
        ranks[m] = ranks[m].mean()
    pos = labels.bool()
    n1, n0 = pos.sum().double(), (~pos).sum().double()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# P19 ------------------------------------------------------------------
def embedding_bag(weight, ids, offsets, mode="mean"):
    """ids 1-D, offsets marks the start of each bag (like nn.EmbeddingBag). Empty bags -> zeros."""
    n_bags = len(offsets)
    lengths = torch.diff(torch.cat([offsets, torch.tensor([len(ids)])]))
    bag_of = torch.repeat_interleave(torch.arange(n_bags), lengths)
    out = torch.zeros(n_bags, weight.shape[1], dtype=weight.dtype).index_add_(0, bag_of, weight[ids])
    if mode == "mean":
        out = out / lengths.clamp(min=1).unsqueeze(1).to(weight.dtype)
    return out


# P20 ------------------------------------------------------------------
def cosine_topk(queries, corpus, k):
    q = queries / queries.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    c = corpus / corpus.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return (q @ c.T).topk(k, -1)


# ======================================================================= tests
def close(a, b, tol=1e-6):
    assert torch.allclose(a, b, atol=tol, rtol=tol), (a - b).abs().max()


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

    z = torch.randn(50, dtype=dt) * 30; y = torch.randint(0, 2, (50,)).to(dt); pw = torch.tensor(3.0, dtype=dt)
    close(bce_with_logits(z, y), F.binary_cross_entropy_with_logits(z, y))
    close(bce_with_logits(z, y, pw), F.binary_cross_entropy_with_logits(z, y, pos_weight=pw))
    print("P03 bce_with_logits ok")

    a, b = torch.randn(5, 8, dtype=dt), torch.randn(5, 8, dtype=dt)
    close(kl_div_logits(a, b), F.kl_div(torch.log_softmax(b, -1), torch.log_softmax(a, -1), log_target=True, reduction="batchmean"))
    print("P04 kl_div ok")

    w, bb = torch.randn(10, dtype=dt), torch.randn(10, dtype=dt)
    close(layer_norm(x, w, bb), F.layer_norm(x, (10,), w, bb))
    close(rms_norm(x, w), F.rms_norm(x, (10,), w, eps=1e-6))
    print("P05 layer_norm/rms_norm ok")

    xb = torch.randn(8, 3, 5, 5, dtype=dt)
    rm, rv, rm2, rv2 = torch.zeros(3, dtype=dt), torch.ones(3, dtype=dt), torch.zeros(3, dtype=dt), torch.ones(3, dtype=dt)
    g3, b3 = torch.randn(3, dtype=dt), torch.randn(3, dtype=dt)
    close(batch_norm(xb, rm, rv, g3, b3, True), F.batch_norm(xb, rm2, rv2, g3, b3, training=True))
    close(rm, rm2); close(rv, rv2)
    close(batch_norm(xb, rm, rv, g3, b3, False), F.batch_norm(xb, rm2, rv2, g3, b3, training=False))
    print("P06 batch_norm ok (incl. running stats)")

    big = torch.ones(200000, dtype=dt)
    d = dropout(big, 0.3, True)
    assert abs((d == 0).double().mean().item() - 0.3) < 0.01 and abs(d.mean().item() - 1) < 0.01
    assert torch.equal(dropout(big, 0.3, False), big)
    print("P07 dropout ok (drop rate, E[x] preserved, eval identity)")

    close(gelu(x), F.gelu(x)); close(gelu(x, "tanh"), F.gelu(x, approximate="tanh")); close(silu(x), F.silu(x))
    D, H = 10, 16
    wg, wu, wd_ = (torch.randn(D, H, dtype=dt), torch.randn(D, H, dtype=dt), torch.randn(H, D, dtype=dt))
    close(swiglu(x, wg, wu, wd_), (F.silu(x @ wg) * (x @ wu)) @ wd_)
    print("P08 gelu/silu/swiglu ok")

    q, k, v = (torch.randn(2, 3, 5, 8, dtype=dt), torch.randn(2, 3, 7, 8, dtype=dt), torch.randn(2, 3, 7, 8, dtype=dt))
    close(sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v))
    close(sdpa(q, k, v, is_causal=True), F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(5, 7, dtype=torch.bool).tril(2)))
    m = torch.rand(5, 7) > 0.3; m[:, 0] = True
    close(sdpa(q, k, v, attn_mask=m), F.scaled_dot_product_attention(q, k, v, attn_mask=m))
    m[1] = False
    assert sdpa(q, k, v, attn_mask=m)[..., 1, :].abs().max() == 0       # fully masked row -> zeros, not NaN
    print("P09 sdpa ok (causal with T!=S, bool mask, fully masked row)")

    pe = sinusoidal_pe(6, 8)
    assert torch.allclose(pe[0], torch.tensor([0., 1.] * 4, dtype=dt))
    xr = torch.randn(2, 6, 8, dtype=dt)
    r = apply_rope(xr)
    close(r.norm(dim=-1), xr.norm(dim=-1))                    # rotation preserves norm
    qq, kk = torch.randn(1, 1, 8, dtype=dt), torch.randn(1, 1, 8, dtype=dt)
    s1 = (apply_rope(qq, 3) * apply_rope(kk, 1)).sum(); s2 = (apply_rope(qq, 10) * apply_rope(kk, 8)).sum()
    close(s1, s2)                                             # depends only on relative position
    close(apply_rope(xr, 0)[:, 2:], apply_rope(xr[:, 2:], 2))  # offset == slicing
    print("P10 sinusoidal PE / RoPE ok (norm, relative position, offset)")

    xc = torch.randn(2, 3, 9, 8, dtype=dt); wc = torch.randn(4, 3, 3, 2, dtype=dt); bc = torch.randn(4, dtype=dt)
    for s, p in [(1, 0), (2, 1), (3, 2)]:
        close(conv2d(xc, wc, bc, s, p), F.conv2d(xc, wc, bc, stride=s, padding=p))
    print("P11 conv2d ok")

    close(pool2d(xc, 2), F.max_pool2d(xc, 2)); close(pool2d(xc, 3, 2, "avg"), F.avg_pool2d(xc, 3, 2))
    print("P12 pooling ok")

    xl = torch.randn(6, 4, dtype=dt, requires_grad=True); wl = torch.randn(3, 4, dtype=dt, requires_grad=True)
    bl = torch.randn(3, dtype=dt, requires_grad=True); dy = torch.randn(6, 3, dtype=dt)
    (F.linear(xl, wl, bl) * dy).sum().backward()
    dx, dw, db = linear_backward(xl.detach(), wl.detach(), dy)
    close(dx, xl.grad); close(dw, wl.grad); close(db, bl.grad)
    print("P13 linear backward ok")

    p_ref = torch.randn(5, dtype=dt, requires_grad=True); p_me = p_ref.detach().clone()
    opt = torch.optim.AdamW([p_ref], lr=0.1, weight_decay=0.1)
    mm, vv = torch.zeros(5, dtype=dt), torch.zeros(5, dtype=dt)
    for t in range(1, 6):
        g = torch.randn(5, dtype=dt)
        p_ref.grad = g.clone(); opt.step()
        adamw_step(p_me, g, mm, vv, t, lr=0.1, wd=0.1)
    close(p_me, p_ref.detach())
    print("P14 AdamW ok (5 steps vs torch.optim.AdamW)")

    lg = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=dt))
    keep = lambda o: torch.isfinite(o)[0].tolist()
    assert keep(top_k_top_p_filter(lg, k=2)) == [True, True, False, False]
    assert keep(top_k_top_p_filter(lg, p=0.75)) == [True, True, False, False]    # 0.5+0.3 >= 0.75 (avoid exact float boundary)
    assert keep(top_k_top_p_filter(lg, p=0.85)) == [True, True, True, False]
    assert keep(top_k_top_p_filter(lg, p=0.1)) == [True, False, False, False]    # always keep top-1
    print("P15 top-k/top-p ok (boundary cases)")

    im, tx = torch.randn(6, 16, dtype=dt), torch.randn(6, 16, dtype=dt)
    ref_logits = F.normalize(im, dim=-1) @ F.normalize(tx, dim=-1).T / 0.07
    lab = torch.arange(6)
    close(clip_loss(im, tx), (F.cross_entropy(ref_logits, lab) + F.cross_entropy(ref_logits.T, lab)) / 2)
    print("P16 CLIP loss ok")

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10.5], [21, 21, 29, 29]], dtype=dt)
    sc = torch.tensor([0.9, 0.8, 0.7, 0.95, 0.6], dtype=dt)
    close(box_iou(boxes[:1], boxes[1:2]), torch.tensor([[81 / 119]], dtype=dt))
    assert nms(boxes, sc, 0.5).tolist() == [3, 2]
    assert nms(boxes, sc, 0.99).tolist() == [3, 0, 1, 2, 4]
    print("P17 IoU / NMS ok")

    lo = torch.randn(100, 10); tg = torch.randint(0, 10, (100,))
    assert topk_accuracy(lo, tg, 10) == 1 and topk_accuracy(lo, tg, 1) == (lo.argmax(-1) == tg).float().mean()
    cm = confusion_matrix(lo.argmax(-1), tg, 10)
    assert cm.sum() == 100 and cm.trace() == (lo.argmax(-1) == tg).sum()
    sco = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.4]); lbl = torch.tensor([0, 0, 1, 1, 1])
    brute = sum((1.0 if sp > sn else 0.5 if sp == sn else 0.0) for sp in sco[lbl == 1] for sn in sco[lbl == 0]) / 6
    assert abs(roc_auc(sco, lbl).item() - brute) < 1e-12
    print(f"P18 metrics ok (AUC with ties = {brute:.4f})")

    W = torch.randn(10, 4, dtype=dt); ids = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]); offs = torch.tensor([0, 3, 3, 6])
    for mode in ("sum", "mean"):
        close(embedding_bag(W, ids, offs, mode), F.embedding_bag(ids, W, offs, mode=mode))
    print("P19 embedding_bag ok (incl. empty bag)")

    qs, cs = torch.randn(3, 8, dtype=dt), torch.randn(50, 8, dtype=dt)
    vals, idx = cosine_topk(qs, cs, 5)
    ref = F.cosine_similarity(qs[:, None], cs[None], dim=-1).topk(5, -1)
    close(vals, ref.values); assert torch.equal(idx, ref.indices)
    print("P20 cosine top-k ok")
    print("OK")


if __name__ == "__main__":
    test_all()
