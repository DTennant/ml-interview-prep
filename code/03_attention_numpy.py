"""Scaled dot-product + multi-head attention in numpy (causal mask + general attention_mask), checked against torch.

Interview checklist:
- scores = Q K^T / sqrt(d_head); mask BEFORE softmax with -inf (or a large negative number)
- attention_mask: bool (..., Tq, Tk), True = query i may attend to key j; one concept covers everything
- causal mask: position i may attend to j <= i  -> np.tril(ones) is the attention_mask
- shapes: x (B, T, D) -> q,k,v (B, H, T, Dh) with Dh = D // H
- complexity O(B * H * T^2 * Dh) time, O(B * H * T^2) memory for the score matrix
"""
import numpy as np


def softmax(z, axis=-1):
    """Numerically stable softmax along `axis`. Output has the same shape as z."""
    # Subtract the max so the largest exponent is exp(0) -> no overflow.
    z_max = z.max(axis=axis, keepdims=True)
    z_shifted = z - z_max
    exp_z = np.exp(z_shifted)
    exp_sum = exp_z.sum(axis=axis, keepdims=True)
    return exp_z / exp_sum


def scaled_dot_product_attention(q, k, v, attention_mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d), blocked entries -> -inf) V

    Input:
        q    (..., Tq, d)    queries
        k    (..., Tk, d)    keys
        v    (..., Tk, dv)   values
        attention_mask  optional bool, broadcastable to (..., Tq, Tk);
                        attention_mask[..., i, j] = True means query i may attend to key j
    Output:
        out  (..., Tq, dv)   weighted sum of values
        attn (..., Tq, Tk)   attention weights, each row sums to 1
    """
    d = q.shape[-1]

    # Step 1: similarity of every query with every key.
    k_transposed = np.swapaxes(k, -1, -2)               # (..., d, Tk)
    scores = q @ k_transposed                           # (..., Tq, Tk)

    # Step 2: scale by 1/sqrt(d). Without it the dot products grow like sqrt(d),
    # softmax saturates and gradients vanish.
    scores = scores / np.sqrt(d)                        # (..., Tq, Tk)

    # Step 3: apply attention_mask BEFORE softmax. Blocked pairs get a huge negative
    # score -> exp(...) = 0 -> zero weight. Each query row must allow at least one key.
    if attention_mask is not None:
        scores = np.where(attention_mask, scores, -1e30)    # (..., Tq, Tk)

    # Step 4: turn scores into probabilities over the keys (last axis).
    attn = softmax(scores, axis=-1)                     # (..., Tq, Tk)

    # Step 5: weighted average of the values.
    out = attn @ v                                      # (..., Tq, dv)
    return out, attn


def causal_mask(T):
    """Causal attention_mask: query i may only attend to keys j <= i (no peeking ahead).

    Output: (T, T) bool, True = query may attend to key. Just a special attention_mask;
    combine it with any other attention_mask using `&`.
    """
    ones = np.ones((T, T), dtype=bool)                  # (T, T)
    mask = np.tril(ones)                                # (T, T)
    return mask


class MultiHeadAttention:
    """Multi-head self-attention (no biases).

    x (B, T, D) -> project to Q, K, V (B, T, D) -> split into H heads (B, H, T, Dh)
    -> attention per head -> merge heads (B, T, D) -> output projection Wo.
    """

    def __init__(self, d_model, n_heads, seed=0):
        """Create the four (D, D) projection matrices Wq, Wk, Wv, Wo."""
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        rng = np.random.default_rng(seed)
        std = 1 / np.sqrt(d_model)

        # Drawn in this order: Wq, Wk, Wv, Wo.
        self.Wq = rng.normal(0, std, (d_model, d_model))    # (D, D)
        self.Wk = rng.normal(0, std, (d_model, d_model))    # (D, D)
        self.Wv = rng.normal(0, std, (d_model, d_model))    # (D, D)
        self.Wo = rng.normal(0, std, (d_model, d_model))    # (D, D)

        self.h = n_heads                     # number of heads H
        self.dh = d_model // n_heads         # per-head dim Dh

    def split(self, x):
        """Split the feature dim into heads: (B, T, D) -> (B, H, T, Dh)."""
        B, T, _ = x.shape
        x = x.reshape(B, T, self.h, self.dh)            # (B, T, H, Dh)
        x = x.transpose(0, 2, 1, 3)                     # (B, H, T, Dh)  heads become a batch dim
        return x

    def merge(self, x):
        """Inverse of split: (B, H, T, Dh) -> (B, T, D) by concatenating the heads."""
        B, H, T, Dh = x.shape
        x = x.transpose(0, 2, 1, 3)                     # (B, T, H, Dh)
        x = x.reshape(B, T, H * Dh)                     # (B, T, D)
        return x

    def __call__(self, x, attention_mask=None):
        """Forward pass.

        Input:  x (B, T, D)
                attention_mask optional bool, broadcastable to (B, H, T, T); True = query may attend to key
        Output: out (B, T, D), attn (B, H, T, T)
        """
        # Step 1: linear projections to queries, keys, values.
        q_full = x @ self.Wq                            # (B, T, D)
        k_full = x @ self.Wk                            # (B, T, D)
        v_full = x @ self.Wv                            # (B, T, D)

        # Step 2: split into H heads, each of size Dh.
        q = self.split(q_full)                          # (B, H, T, Dh)
        k = self.split(k_full)                          # (B, H, T, Dh)
        v = self.split(v_full)                          # (B, H, T, Dh)

        # Step 3: attention inside each head independently.
        head_out, attn = scaled_dot_product_attention(q, k, v, attention_mask)   # (B, H, T, Dh), (B, H, T, T)

        # Step 4: concatenate heads back together.
        merged = self.merge(head_out)                   # (B, T, D)

        # Step 5: output projection mixes information across heads.
        out = merged @ self.Wo                          # (B, T, D)
        return out, attn


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 6, 16, 4
    x = rng.normal(size=(B, T, D))
    mha = MultiHeadAttention(D, H)
    # attention_mask: (B, 1, T, T), True = query i may attend to key j (broadcast over heads).
    # Batch 0: plain causal. Batch 1: causal AND keys 4, 5 are blocked for every query.
    attention_mask = np.zeros((B, 1, T, T), dtype=bool)
    attention_mask[0, 0] = causal_mask(T)
    attention_mask[1, 0] = causal_mask(T)
    attention_mask[1, 0, :, 4:] = False
    out, attn = mha(x, attention_mask)
    print("out", out.shape, "attn", attn.shape)

    # sanity: causal -> upper triangle of attention is zero; rows sum to 1
    assert np.allclose(np.triu(attn[0, 0], 1), 0)
    assert np.allclose(attn.sum(-1), 1)
    # sanity: blocked keys (positions 4,5 of batch 1) get zero weight
    assert np.allclose(attn[1, :, :, 4:], 0)

    try:
        import torch
        import torch.nn.functional as F
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        q, k, v = (t(mha.split(x @ W)) for W in (mha.Wq, mha.Wk, mha.Wv))
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor(attention_mask))  # torch bool mask: True = may attend
        ref = t(mha.merge(ref.numpy())) @ t(mha.Wo)
        print("max diff vs torch:", (ref - t(out)).abs().max().item())
        assert torch.allclose(ref, t(out), atol=1e-8)
    except ImportError:
        print("torch not installed, skipped reference check")
    print("OK")
