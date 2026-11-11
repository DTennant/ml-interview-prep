"""Multi-head attention, KV cache, GQA / MQA from scratch in PyTorch (no nn.MultiheadAttention, no SDPA in the impl).

One module covers all three variants through n_kv_heads:
    n_kv_heads == n_heads      -> MHA
    1 < n_kv_heads < n_heads   -> GQA (each KV head is shared by n_heads // n_kv_heads query heads)
    n_kv_heads == 1            -> MQA

KV cache: pre-allocated (B, n_kv_heads, max_T, Dh) buffers + a write position, like real inference engines.
  prefill: run the whole prompt once, write K/V for positions [0, T)
  decode : one token at a time, write K/V at position t, attend over [0, t]

Tests at the bottom:
  1. MHA forward == torch SDPA reference
  2. GQA forward == torch SDPA(enable_gqa=True) reference, and == "repeat_interleave K/V then MHA"
  3. prefill + step-by-step decode with cache == full recompute without cache (MHA, GQA, MQA)
  4. chunked prefill (prompt fed in two chunks) == single prefill
  5. KV cache memory: GQA/MQA shrink it by n_heads / n_kv_heads

Interview checklist:
- shapes: x (B,T,D) -> q (B,H,T,Dh), k/v (B,Hkv,T,Dh); Dh = D // H; group size G = H // Hkv
- GQA trick: reshape q to (B,Hkv,G,T,Dh) and broadcast k/v as (B,Hkv,1,S,Dh) -- no copy. repeat_interleave also works
  but materialises H/Hkv copies of K/V (fine to say "this is what I'd optimise")
- causal mask with a cache: query i (absolute position pos+i) may see keys j <= pos+i -> tril(ones(T,S), diagonal=S-T)
- RoPE must use absolute positions (offset by the cache position), applied to q and k BEFORE writing k to the cache
- cache memory per token = 2 * L * Hkv * Dh * bytes; decode is memory-bandwidth bound, so smaller KV = faster decode
- why GQA: MQA loses quality; GQA keeps most of the quality of MHA with close to MQA's KV size (Llama-2-70B: 64 q heads, 8 kv heads)
- complexity per decode step: O(S * D) compute, reading S * 2 * Hkv * Dh values from memory
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================================
# RoPE helpers
# =====================================================================================================
def rope_angles(max_T, dh, base=10000.0):
    """Pre-compute RoPE cos / sin tables for every position up to max_T.

    Each head vector is viewed as dh/2 pairs; pair i at position t is rotated by angle t * theta_i,
    with theta_i = base^(-2i/dh).

    Input:  max_T (int), dh head dimension (even)
    Output: cos, sin  each (max_T, dh/2), float64
    """
    # Step 1: one rotation frequency per pair of dimensions
    pair_index_times_2 = torch.arange(0, dh, 2, dtype=torch.float64)       # (dh/2,)
    inv_freq = 1.0 / (base ** (pair_index_times_2 / dh))                   # (dh/2,)

    # Step 2: angle = position * frequency, for all combinations
    positions = torch.arange(max_T, dtype=torch.float64)                   # (max_T,)
    angles = torch.outer(positions, inv_freq)                              # (max_T, dh/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each consecutive pair (x0,x1), (x2,x3), ... of the last dim by its position's angle.

    Input:  x (..., T, dh);  cos, sin (T, dh/2) for exactly those T positions
    Output: (..., T, dh) same shape as x
    """
    # Step 1: split every pair into its first (a) and second (b) element
    x_even = x[..., 0::2]                                                  # (..., T, dh/2)
    x_odd = x[..., 1::2]                                                   # (..., T, dh/2)

    # Step 2: 2D rotation  (a, b) -> (a cos - b sin, a sin + b cos)
    rotated_even = x_even * cos - x_odd * sin                              # (..., T, dh/2)
    rotated_odd = x_even * sin + x_odd * cos                               # (..., T, dh/2)

    # Step 3: interleave back to (a0', b0', a1', b1', ...)
    stacked = torch.stack([rotated_even, rotated_odd], dim=-1)             # (..., T, dh/2, 2)
    return stacked.flatten(-2)                                             # (..., T, dh)


# =====================================================================================================
# KV cache
# =====================================================================================================
class KVCache:
    """Pre-allocated key/value cache for ONE attention layer.

    Real inference engines allocate the whole buffer up front (no torch.cat every step, which would re-copy the
    entire cache each token) and keep a write pointer `pos`.

    Buffers: k, v  each (B, Hkv, max_T, Dh); only [:, :, :pos] holds valid data.
    """

    def __init__(self, B, n_kv_heads, max_T, dh, dtype=torch.float64):
        self.k = torch.zeros(B, n_kv_heads, max_T, dh, dtype=dtype)       # (B, Hkv, max_T, Dh)
        self.v = torch.zeros(B, n_kv_heads, max_T, dh, dtype=dtype)       # (B, Hkv, max_T, Dh)
        self.pos = 0   # number of tokens already cached == absolute position of the next token

    def update(self, k, v):
        """Write the new tokens' keys/values at [pos, pos + T) and advance the write pointer.

        Input:  k, v (B, Hkv, T, Dh) for the T new tokens
        Output: views k_all, v_all (B, Hkv, pos + T, Dh) covering ALL cached tokens (old + new)
        """
        num_new = k.shape[2]
        start = self.pos
        end = self.pos + num_new
        assert end <= self.k.shape[2], "cache overflow"

        # Step 1: write into the pre-allocated buffers (in place, no reallocation)
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v

        # Step 2: advance the pointer
        self.pos = end

        # Step 3: return only the valid prefix (views, no copy)
        k_all = self.k[:, :, :self.pos]                                    # (B, Hkv, pos, Dh)
        v_all = self.v[:, :, :self.pos]                                    # (B, Hkv, pos, Dh)
        return k_all, v_all

    def nbytes(self):
        """Total memory of the k and v buffers in bytes."""
        num_elements = self.k.numel() + self.v.numel()
        return self.k.element_size() * num_elements


# =====================================================================================================
# Attention (MHA / GQA / MQA)
# =====================================================================================================
class Attention(nn.Module):
    """Causal self-attention supporting MHA, GQA and MQA via n_kv_heads, with optional RoPE and KV cache.

    Input:  x (B, T, D); cache: KVCache or None
    Output: (B, T, D)
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None, max_T=512, rope=True):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads          # default: plain MHA
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0  # every KV head must serve the same number of query heads

        self.h = n_heads                  # number of query heads (H)
        self.hkv = n_kv_heads             # number of key/value heads (Hkv)
        self.dh = d_model // n_heads      # head dimension (Dh)
        self.g = n_heads // n_kv_heads    # group size G: query heads per KV head

        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)       # D -> H * Dh
        # K and V projections are SMALLER than Q for GQA/MQA: only Hkv heads -> fewer params and a smaller cache
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)    # D -> Hkv * Dh
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)    # D -> Hkv * Dh
        self.wo = nn.Linear(n_heads * self.dh, d_model, bias=False)       # H * Dh -> D

        self.rope = rope
        cos, sin = rope_angles(max_T, self.dh)                            # each (max_T, Dh/2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def project(self, x, pos):
        """Compute q, k, v for x and apply RoPE at absolute positions pos .. pos + T - 1.

        Input:  x (B, T, D); pos = absolute position of x[:, 0] (= number of tokens already in the cache)
        Output: q (B, H, T, Dh), k (B, Hkv, T, Dh), v (B, Hkv, T, Dh)
        """
        B, T, _ = x.shape

        # Step 1: linear projections, split into heads, move the head axis before time
        q = self.wq(x)                                                     # (B, T, H * Dh)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)                  # (B, H, T, Dh)
        k = self.wk(x)                                                     # (B, T, Hkv * Dh)
        k = k.view(B, T, self.hkv, self.dh).transpose(1, 2)                # (B, Hkv, T, Dh)
        v = self.wv(x)                                                     # (B, T, Hkv * Dh)
        v = v.view(B, T, self.hkv, self.dh).transpose(1, 2)                # (B, Hkv, T, Dh)

        # Step 2: RoPE on q and k using ABSOLUTE positions (offset by pos). This happens before k is written to
        # the cache, so cached keys already carry their absolute position and are never rotated again.
        if self.rope:
            cos = self.cos[pos:pos + T]                                    # (T, Dh/2)
            sin = self.sin[pos:pos + T]                                    # (T, Dh/2)
            q = apply_rope(q, cos, sin)                                    # (B, H, T, Dh)
            k = apply_rope(k, cos, sin)                                    # (B, Hkv, T, Dh)
        return q, k, v

    @staticmethod
    def attend(q, k, v, g):
        """Causal grouped attention. The T queries are aligned to the END of the S keys.

        GQA trick: instead of copying each KV head G times, reshape the H query heads into (Hkv, G) and let
        broadcasting share one KV head across its G query heads -- no extra memory.

        Input:  q (B, H, T, Dh); k, v (B, Hkv, S, Dh); g = H // Hkv
        Output: (B, H, T, Dh)
        """
        B, H, T, Dh = q.shape
        S = k.shape[2]
        num_kv_heads = H // g

        # Step 1: group the query heads by the KV head they share
        q_grouped = q.view(B, num_kv_heads, g, T, Dh)                      # (B, Hkv, G, T, Dh)

        # Step 2: add a size-1 group axis to k/v so they broadcast over the G query heads (no copy)
        k_shared = k.unsqueeze(2)                                          # (B, Hkv, 1, S, Dh)
        v_shared = v.unsqueeze(2)                                          # (B, Hkv, 1, S, Dh)

        # Step 3: scaled dot-product scores
        scores = q_grouped @ k_shared.transpose(-1, -2)                    # (B, Hkv, G, T, S)
        scores = scores / math.sqrt(Dh)                                    # (B, Hkv, G, T, S)

        # Step 4: attention_mask -- a bool tensor (T_q, T_k), True = this query may attend to this key.
        # Here it is just the causal mask: query i is at absolute position (S - T + i), so it may see keys
        # j <= S - T + i -> tril with diagonal offset S - T (plain lower triangle when S == T, all True for
        # 1 decode query). Shape (T, S) broadcasts over (B, Hkv, G).
        attention_mask = torch.ones(T, S, dtype=torch.bool, device=q.device)
        attention_mask = attention_mask.tril(diagonal=S - T)               # (T, S)
        # Where the mask is False, set the score to -inf so softmax gives that key weight 0.
        scores = scores.masked_fill(~attention_mask, float("-inf"))        # (B, Hkv, G, T, S)

        # Step 5: softmax over keys, weighted sum of values
        weights = scores.softmax(dim=-1)                                   # (B, Hkv, G, T, S)
        out_grouped = weights @ v_shared                                   # (B, Hkv, G, T, Dh)

        # Step 6: merge (Hkv, G) back into H query heads
        return out_grouped.reshape(B, H, T, Dh)                            # (B, H, T, Dh)

    def forward(self, x, cache: KVCache = None):
        """Full attention layer.

        Input:  x (B, T, D) -- the whole sequence (no cache), a prompt chunk (prefill) or 1 token (decode)
                cache: KVCache or None
        Output: (B, T, D)
        """
        B, T, _ = x.shape

        # Step 1: the absolute position of x[:, 0] is how many tokens are already cached
        if cache is None:
            pos = 0
        else:
            pos = cache.pos

        # Step 2: project + RoPE
        q, k, v = self.project(x, pos)          # q (B, H, T, Dh); k, v (B, Hkv, T, Dh)

        # Step 3: store the new k/v and get back k/v for ALL tokens so far
        if cache is not None:
            k, v = cache.update(k, v)           # k, v (B, Hkv, pos + T, Dh)

        # Step 4: attention
        out = self.attend(q, k, v, self.g)                                 # (B, H, T, Dh)

        # Step 5: concatenate heads and project back to d_model
        out = out.transpose(1, 2)                                          # (B, T, H, Dh)
        out = out.reshape(B, T, self.h * self.dh)                          # (B, T, H * Dh)
        return self.wo(out)                                                # (B, T, D)


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    """KV cache size in bytes for a whole model.

    Factor 2 because we store both K and V; per token per layer that's Hkv * Dh values each.
    bytes_per = 2 for bf16/fp16.
    """
    k_and_v = 2
    values_per_token_per_layer = n_kv_heads * head_dim
    total_values = k_and_v * n_layers * values_per_token_per_layer * seq_len * batch
    return total_values * bytes_per


# ---------------------------------------------------------------- tests
if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.float64
    B, T, D = 2, 10, 64

    # 1 + 2: forward vs torch SDPA reference (rope off so the reference is simple)
    for name, H, Hkv in [("MHA", 8, 8), ("GQA", 8, 2), ("MQA", 8, 1)]:
        attn = Attention(D, H, Hkv, rope=False).to(dt)
        x = torch.randn(B, T, D, dtype=dt)
        out = attn(x)
        q, k, v = attn.project(x, 0)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(H != Hkv))
        ref = attn.wo(ref.transpose(1, 2).reshape(B, T, D))
        assert torch.allclose(out, ref, atol=1e-10), name

        # GQA == MHA after explicitly repeating each KV head G times
        group = H // Hkv
        k_repeated = k.repeat_interleave(group, dim=1)                     # (B, H, T, Dh)
        v_repeated = v.repeat_interleave(group, dim=1)                     # (B, H, T, Dh)
        ref2 = F.scaled_dot_product_attention(q, k_repeated, v_repeated, is_causal=True)
        ref2 = attn.wo(ref2.transpose(1, 2).reshape(B, T, D))
        assert torch.allclose(out, ref2, atol=1e-10), name
        print(f"{name}: forward matches torch SDPA (max diff {(out - ref).abs().max():.1e}); "
              f"wk params {attn.wk.weight.numel()} vs wq {attn.wq.weight.numel()}")

    # 3: prefill + decode with cache == full recompute (rope on)
    for name, H, Hkv in [("MHA", 8, 8), ("GQA", 8, 2), ("MQA", 8, 1)]:
        attn = Attention(D, H, Hkv, rope=True).to(dt)
        x = torch.randn(B, T, D, dtype=dt)
        full = attn(x)                                                    # no cache, causal over all T
        cache = KVCache(B, Hkv, 64, D // H)
        prompt_len = 4
        outs = [attn(x[:, :prompt_len], cache)]                           # prefill
        for t in range(prompt_len, T):                                    # decode one token at a time
            outs.append(attn(x[:, t:t + 1], cache))
        cached = torch.cat(outs, 1)
        assert torch.allclose(full, cached, atol=1e-10), name
        print(f"{name}: prefill({prompt_len}) + {T - prompt_len} decode steps == full recompute "
              f"(max diff {(full - cached).abs().max():.1e}); cache {cache.nbytes()} bytes")

    # 4: chunked prefill
    attn = Attention(D, 8, 2).to(dt)
    x = torch.randn(B, T, D, dtype=dt)
    cache_single = KVCache(B, 2, 64, 8)
    cache_chunked = KVCache(B, 2, 64, 8)
    one = attn(x, cache_single)
    first_chunk = attn(x[:, :6], cache_chunked)
    second_chunk = attn(x[:, 6:], cache_chunked)
    two = torch.cat([first_chunk, second_chunk], 1)
    assert torch.allclose(one, two, atol=1e-10)
    print("chunked prefill (6 + 4) == single prefill")

    # 5: KV cache size, Llama-2-70B-like shapes, bf16, 4k context
    mha = kv_cache_bytes(80, 64, 128, 4096)
    gqa = kv_cache_bytes(80, 8, 128, 4096)
    mqa = kv_cache_bytes(80, 1, 128, 4096)
    print(f"KV cache @4k, 80 layers, bf16: MHA {mha / 2**30:.1f} GiB | GQA(8) {gqa / 2**30:.2f} GiB | MQA {mqa / 2**30:.3f} GiB")
    assert mha == 8 * gqa == 64 * mqa
    print("OK")
