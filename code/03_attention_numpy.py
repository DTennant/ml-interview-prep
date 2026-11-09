"""Scaled dot-product + multi-head attention in numpy (causal + padding masks), checked against torch.

Interview checklist:
- scores = Q K^T / sqrt(d_head); mask BEFORE softmax with -inf (or a large negative number)
- causal mask: position i may attend to j <= i  -> np.triu(ones, k=1) are the masked entries
- shapes: x (B, T, D) -> q,k,v (B, H, T, Dh) with Dh = D // H
- complexity O(B * H * T^2 * Dh) time, O(B * H * T^2) memory for the score matrix
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q (..., Tq, d), k (..., Tk, d), v (..., Tk, dv); mask broadcastable to (..., Tq, Tk), True = keep."""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e30)
    attn = softmax(scores, -1)
    return attn @ v, attn


def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))      # (T, T), True = allowed


def padding_mask(lengths, T):
    """lengths (B,) -> (B, 1, 1, T) mask over keys."""
    return (np.arange(T)[None, :] < np.asarray(lengths)[:, None])[:, None, None, :]


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(0, s, (d_model, d_model)) for _ in range(4))
        self.h, self.dh = n_heads, d_model // n_heads

    def split(self, x):          # (B, T, D) -> (B, H, T, Dh)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def merge(self, x):          # (B, H, T, Dh) -> (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def __call__(self, x, mask=None):
        q, k, v = self.split(x @ self.Wq), self.split(x @ self.Wk), self.split(x @ self.Wv)
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        return self.merge(out) @ self.Wo, attn


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 6, 16, 4
    x = rng.normal(size=(B, T, D))
    mha = MultiHeadAttention(D, H)
    mask = causal_mask(T)[None, None] & padding_mask([6, 4], T)
    out, attn = mha(x, mask)
    print("out", out.shape, "attn", attn.shape)

    # sanity: causal -> upper triangle of attention is zero; rows sum to 1
    assert np.allclose(np.triu(attn[0, 0], 1), 0)
    assert np.allclose(attn.sum(-1), 1)
    # sanity: padded keys (positions 4,5 of batch 1) get zero weight
    assert np.allclose(attn[1, :, :, 4:], 0)

    try:
        import torch
        import torch.nn.functional as F
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        q, k, v = (t(mha.split(x @ W)) for W in (mha.Wq, mha.Wk, mha.Wv))
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor(mask))
        ref = t(mha.merge(ref.numpy())) @ t(mha.Wo)
        # rows that are fully valid should match exactly
        print("max diff vs torch:", (ref - t(out)).abs().max().item())
        assert torch.allclose(ref, t(out), atol=1e-8)
    except ImportError:
        print("torch not installed, skipped reference check")
    print("OK")
