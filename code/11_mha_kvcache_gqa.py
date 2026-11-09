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


def rope_angles(max_T, dh, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh)
    ang = torch.outer(torch.arange(max_T, dtype=torch.float64), inv)       # (max_T, dh/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (..., T, dh); cos/sin (T, dh/2). Rotates (x0,x1), (x2,x3), ..."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class KVCache:
    def __init__(self, B, n_kv_heads, max_T, dh, dtype=torch.float64):
        self.k = torch.zeros(B, n_kv_heads, max_T, dh, dtype=dtype)
        self.v = torch.zeros(B, n_kv_heads, max_T, dh, dtype=dtype)
        self.pos = 0                                                      # number of tokens already cached

    def update(self, k, v):
        """k, v: (B, Hkv, T, Dh) for the new tokens. Returns views over all cached tokens."""
        T = k.shape[2]
        assert self.pos + T <= self.k.shape[2], "cache overflow"
        self.k[:, :, self.pos:self.pos + T] = k
        self.v[:, :, self.pos:self.pos + T] = v
        self.pos += T
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]

    def nbytes(self):
        return self.k.element_size() * (self.k.numel() + self.v.numel())


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, max_T=512, rope=True):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.hkv, self.dh = n_heads, n_kv_heads, d_model // n_heads
        self.g = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)   # smaller than wq for GQA/MQA
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d_model, bias=False)
        self.rope = rope
        cos, sin = rope_angles(max_T, self.dh)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def project(self, x, pos):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)       # (B, H,   T, Dh)
        k = self.wk(x).view(B, T, self.hkv, self.dh).transpose(1, 2)     # (B, Hkv, T, Dh)
        v = self.wv(x).view(B, T, self.hkv, self.dh).transpose(1, 2)
        if self.rope:                                                     # absolute positions pos..pos+T-1
            cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    @staticmethod
    def attend(q, k, v, g):
        """q (B,H,T,Dh), k/v (B,Hkv,S,Dh) -> (B,H,T,Dh). Causal, aligned to the END of the key sequence."""
        B, H, T, Dh = q.shape
        S = k.shape[2]
        q = q.view(B, H // g, g, T, Dh)                                   # group query heads per KV head
        k, v = k.unsqueeze(2), v.unsqueeze(2)                             # (B, Hkv, 1, S, Dh), broadcast, no copy
        scores = q @ k.transpose(-1, -2) / math.sqrt(Dh)                  # (B, Hkv, G, T, S)
        mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(-1)
        return (w @ v).reshape(B, H, T, Dh)

    def forward(self, x, cache: KVCache = None):
        B, T, _ = x.shape
        pos = 0 if cache is None else cache.pos
        q, k, v = self.project(x, pos)
        if cache is not None:
            k, v = cache.update(k, v)                                     # k/v now cover [0, pos+T)
        out = self.attend(q, k, v, self.g)
        return self.wo(out.transpose(1, 2).reshape(B, T, self.h * self.dh))


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per


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
        kr, vr = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
        ref2 = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
        assert torch.allclose(out, attn.wo(ref2.transpose(1, 2).reshape(B, T, D)), atol=1e-10), name
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
    c1, c2 = KVCache(B, 2, 64, 8), KVCache(B, 2, 64, 8)
    one = attn(x, c1)
    two = torch.cat([attn(x[:, :6], c2), attn(x[:, 6:], c2)], 1)
    assert torch.allclose(one, two, atol=1e-10)
    print("chunked prefill (6 + 4) == single prefill")

    # 5: KV cache size, Llama-2-70B-like shapes, bf16, 4k context
    mha = kv_cache_bytes(80, 64, 128, 4096)
    gqa = kv_cache_bytes(80, 8, 128, 4096)
    mqa = kv_cache_bytes(80, 1, 128, 4096)
    print(f"KV cache @4k, 80 layers, bf16: MHA {mha / 2**30:.1f} GiB | GQA(8) {gqa / 2**30:.2f} GiB | MQA {mqa / 2**30:.3f} GiB")
    assert mha == 8 * gqa == 64 * mqa
    print("OK")
