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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# =====================================================================================================
# RMSNorm
# =====================================================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (used by Llama instead of LayerNorm).

    Unlike LayerNorm it does NOT subtract the mean and has no bias; it only rescales each vector so that its
    root-mean-square is 1, then multiplies by a learned per-channel gain.

        y = x / sqrt(mean(x^2) + eps) * w

    Input:  x (..., D)
    Output: y (..., D)   same shape
    """

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))   # learned gain, starts at 1 so the norm is initially "pure"
        self.eps = eps                         # avoids division by zero for an all-zero vector

    def forward(self, x):
        # Step 1: mean of squares over the feature dimension
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)             # (..., 1)

        # Step 2: 1 / RMS  (rsqrt = 1 / sqrt, one fused op)
        inv_rms = torch.rsqrt(mean_square + self.eps)                 # (..., 1)

        # Step 3: normalise, then apply the learned per-channel gain
        x_normed = x * inv_rms                                        # (..., D)
        return x_normed * self.w                                      # (..., D)


# =====================================================================================================
# Rotary position embedding (RoPE)
# =====================================================================================================
def rope_cache(T, dh, base=10000.0):
    """Pre-compute the cos / sin tables used by RoPE.

    RoPE treats each head vector as dh/2 pairs of numbers (x0,x1), (x2,x3), ... and rotates pair i by the angle
    position * theta_i, where theta_i = base^(-2i/dh). Low i -> fast rotation (fine position info),
    high i -> slow rotation (coarse / long-range position info).

    Input:  T   max sequence length, dh head dimension (must be even)
    Output: cos, sin  each (T, dh/2); entry [t, i] = cos/sin(t * theta_i)
    """
    # Step 1: one frequency per pair of dimensions
    pair_index_times_2 = torch.arange(0, dh, 2).float()                # (dh/2,)  = 0, 2, 4, ...
    inv_freq = 1.0 / (base ** (pair_index_times_2 / dh))               # (dh/2,)  theta_i

    # Step 2: angle for every (position, frequency) combination
    positions = torch.arange(T).float()                                # (T,)
    angles = torch.outer(positions, inv_freq)                          # (T, dh/2)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate every consecutive pair (x0,x1), (x2,x3), ... of the last dimension by its position's angle.

    2D rotation of a pair (a, b) by angle phi:
        a' = a * cos(phi) - b * sin(phi)
        b' = a * sin(phi) + b * cos(phi)

    Input:  x (B, H, T, dh);  cos, sin (T, dh/2) -- already sliced to the T positions of x
    Output: (B, H, T, dh) rotated, same shape as x
    """
    # Step 1: split into the "first" and "second" element of every pair
    x_even = x[..., 0::2]                                              # (B, H, T, dh/2)  a's
    x_odd = x[..., 1::2]                                               # (B, H, T, dh/2)  b's

    # Step 2: apply the 2D rotation to every pair (cos/sin broadcast over B and H)
    rotated_even = x_even * cos - x_odd * sin                          # (B, H, T, dh/2)
    rotated_odd = x_even * sin + x_odd * cos                           # (B, H, T, dh/2)

    # Step 3: interleave them back into the original layout (a0', b0', a1', b1', ...)
    stacked = torch.stack([rotated_even, rotated_odd], dim=-1)         # (B, H, T, dh/2, 2)
    return stacked.flatten(-2)                                         # (B, H, T, dh)


# =====================================================================================================
# Causal multi-head self-attention with optional KV cache
# =====================================================================================================
class Attention(nn.Module):
    """Causal multi-head self-attention with RoPE and an optional KV cache.

    Input:  x (B, T, D); cos, sin (T, Dh/2) for the absolute positions of these T tokens;
            cache: None (no caching) or a dict that this layer reads/writes keys "k", "v" of shape (B, H, T_past, Dh)
    Output: (B, T, D)
    """

    def __init__(self, d, h):
        super().__init__()
        self.h = h              # number of heads
        self.dh = d // h        # dimension per head
        # One fused matmul produces q, k and v at once (3 * d outputs) -- faster than three separate Linears.
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        # Output projection that mixes the heads back together.
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape

        # Step 1: project to q, k, v and split into heads.
        qkv = self.qkv(x)                                              # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.h, self.dh)                       # (B, T, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)                               # (3, B, H, T, Dh)
        q = qkv[0]                                                     # (B, H, T, Dh)
        k = qkv[1]                                                     # (B, H, T, Dh)
        v = qkv[2]                                                     # (B, H, T, Dh)

        # Step 2: add position information with RoPE (only q and k; v carries content, not position).
        # Done BEFORE caching, so the cached keys already carry their absolute position and never need re-rotating.
        q = apply_rope(q, cos, sin)                                    # (B, H, T, Dh)
        k = apply_rope(k, cos, sin)                                    # (B, H, T, Dh)

        # Step 3: KV cache. Append the new keys/values to the ones from earlier steps, then store the result.
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)                  # (B, H, T_past + T, Dh)
                v = torch.cat([cache["v"], v], dim=2)                  # (B, H, T_past + T, Dh)
            cache["k"] = k
            cache["v"] = v
        Tk = k.shape[2]   # total number of keys we attend over (= T without a cache)

        # Step 4: attention scores, scaled by 1/sqrt(Dh) so the softmax doesn't saturate as Dh grows.
        scores = q @ k.transpose(-1, -2)                               # (B, H, T, Tk)
        scores = scores / math.sqrt(self.dh)                           # (B, H, T, Tk)

        # Step 5: attention_mask -- a bool tensor (T_q, T_k), True = this query may attend to this key.
        # Here it is just the causal mask: the T queries are the LAST T positions of the Tk keys, so query i
        # (absolute position Tk - T + i) may see keys 0 .. Tk - T + i  ->  tril with diagonal offset Tk - T.
        # Without a cache Tk == T and this is the usual lower triangle; with 1 decode query it is all True.
        attention_mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device)
        attention_mask = attention_mask.tril(diagonal=Tk - T)          # (T, Tk)
        # Where the mask is False, set the score to -inf so softmax gives that key weight 0.
        scores = scores.masked_fill(~attention_mask, float("-inf"))    # (B, H, T, Tk)

        # Step 6: softmax over keys -> attention weights, then weighted sum of the values.
        weights = scores.softmax(dim=-1)                               # (B, H, T, Tk)
        per_head_out = weights @ v                                     # (B, H, T, Dh)

        # Step 7: concatenate heads back into D channels and mix them with the output projection.
        per_head_out = per_head_out.transpose(1, 2)                    # (B, T, H, Dh)
        merged = per_head_out.reshape(B, T, D)                         # (B, T, D)
        return self.o(merged)                                          # (B, T, D)


# =====================================================================================================
# SwiGLU feed-forward
# =====================================================================================================
class SwiGLU(nn.Module):
    """Gated feed-forward block used in Llama / PaLM.

        out = W2( silu(W1 x)  *  W3 x )
                  ^ gate         ^ value

    Input:  x (B, T, D)
    Output: (B, T, D)
    """

    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)    # gate projection
        self.w3 = nn.Linear(d, hidden, bias=False)    # value ("up") projection
        self.w2 = nn.Linear(hidden, d, bias=False)    # down projection back to model width

    def forward(self, x):
        gate = F.silu(self.w1(x))                     # (B, T, hidden)  silu(z) = z * sigmoid(z)
        value = self.w3(x)                            # (B, T, hidden)
        gated = gate * value                          # (B, T, hidden)  elementwise gating
        return self.w2(gated)                         # (B, T, D)


# =====================================================================================================
# Transformer block
# =====================================================================================================
class Block(nn.Module):
    """One pre-norm transformer block:

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    Pre-norm (norm inside the residual branch) keeps the residual stream an identity path, which makes deep
    models train stably without careful warmup tricks.

    Input:  x (B, T, D); cos, sin (T, Dh/2); cache dict or None (passed through to attention)
    Output: (B, T, D)
    """

    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, h)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, 4 * d)

    def forward(self, x, cos, sin, cache=None):
        # Sub-layer 1: attention (tokens exchange information)
        attn_out = self.attn(self.n1(x), cos, sin, cache)              # (B, T, D)
        x = x + attn_out                                               # (B, T, D)

        # Sub-layer 2: MLP (each token processed independently)
        mlp_out = self.mlp(self.n2(x))                                 # (B, T, D)
        x = x + mlp_out                                                # (B, T, D)
        return x


# =====================================================================================================
# Full GPT model
# =====================================================================================================
class GPT(nn.Module):
    """Decoder-only language model: embedding -> L blocks -> final norm -> lm_head (tied to the embedding).

    Input:  idx (B, T) token ids
            caches: None, or a list with one dict per layer (KV cache)
            pos: absolute position of idx[:, 0] (needed when decoding with a cache so RoPE uses the right angles)
    Output: logits (B, T, vocab)
    """

    def __init__(self, vocab, d=64, h=4, L=2, max_T=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

        # Weight tying: the output projection reuses the embedding matrix (both are (vocab, d)).
        # Saves vocab * d parameters and usually helps small models.
        self.head.weight = self.emb.weight

        # RoPE tables computed once for all positions; buffers move with .to(device) but are not saved in state_dict.
        cos, sin = rope_cache(max_T, d // h)                           # each (max_T, Dh/2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, caches=None, pos=0):
        T = idx.shape[1]

        # Step 1: pick the rotation angles for absolute positions pos .. pos + T - 1
        cos = self.cos[pos:pos + T]                                    # (T, Dh/2)
        sin = self.sin[pos:pos + T]                                    # (T, Dh/2)

        # Step 2: token embeddings (no absolute position embedding -- RoPE handles position inside attention)
        x = self.emb(idx)                                              # (B, T, D)

        # Step 3: run the blocks, giving each layer its own cache dict
        for layer_index, block in enumerate(self.blocks):
            if caches is None:
                layer_cache = None
            else:
                layer_cache = caches[layer_index]
            x = block(x, cos, sin, layer_cache)                        # (B, T, D)

        # Step 4: final norm and projection to vocabulary logits
        x = self.norm(x)                                               # (B, T, D)
        logits = self.head(x)                                          # (B, T, vocab)
        return logits


# =====================================================================================================
# Sampling
# =====================================================================================================
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """Pick the next token from the logits of the last position.

    temperature == 0 -> greedy (argmax). Otherwise:
      1. divide logits by temperature (<1 sharper, >1 flatter)
      2. top-k: keep only the k largest logits
      3. top-p (nucleus): keep the smallest set of most-likely tokens whose total probability >= top_p
      4. sample from the softmax of what is left

    Input:  logits (B, V)
    Output: next token ids (B, 1)
    """
    # Greedy decoding: deterministic, used to compare cached vs non-cached generation.
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)                     # (B, 1)

    # Step 1: temperature scaling
    logits = logits / temperature                                      # (B, V)

    # Step 2: top-k filtering -- everything smaller than the k-th largest logit gets -inf (probability 0)
    if top_k is not None:
        top_values = torch.topk(logits, top_k, dim=-1).values          # (B, k) sorted descending
        kth_largest = top_values[..., -1:]                             # (B, 1)
        logits = logits.masked_fill(logits < kth_largest, float("-inf"))   # (B, V)

    # Step 3: top-p (nucleus) filtering
    if top_p is not None:
        # 3a: sort tokens from most to least likely
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)   # each (B, V)
        sorted_probs = sorted_logits.softmax(dim=-1)                   # (B, V)

        # 3b: probability mass of all tokens strictly BEFORE each token in the sorted list
        cumulative_probs = sorted_probs.cumsum(dim=-1)                 # (B, V) mass including this token
        mass_before = cumulative_probs - sorted_probs                  # (B, V) mass excluding this token

        # 3c: drop a token if the tokens before it already reach top_p. This keeps the smallest prefix whose
        #     mass is >= top_p, and always keeps the single most likely token (its mass_before is 0).
        remove = mass_before > top_p                                   # (B, V) bool
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))   # (B, V)

        # 3d: undo the sort -- write each filtered logit back to its original vocabulary index
        unsorted = torch.full_like(logits, float("-inf"))              # (B, V)
        logits = unsorted.scatter(-1, sorted_idx, sorted_logits)       # (B, V)

    # Step 4: sample one token from the remaining distribution
    probs = logits.softmax(dim=-1)                                     # (B, V)
    return torch.multinomial(probs, num_samples=1)                     # (B, 1)


@torch.no_grad()
def generate_naive(model, idx, n, **kw):
    """Generate n tokens WITHOUT a KV cache: every step re-runs the model on the whole sequence so far.

    Simple but O(T^2) work per step. Used as the reference for generate_kv.

    Input:  idx (B, T0) prompt ids; kw are passed to sample_next
    Output: (B, T0 + n)
    """
    for _ in range(n):
        all_logits = model(idx)                                        # (B, T, V)
        last_logits = all_logits[:, -1]                                # (B, V)  only the last position matters
        next_token = sample_next(last_logits, **kw)                    # (B, 1)
        idx = torch.cat([idx, next_token], dim=1)                      # (B, T + 1)
    return idx


@torch.no_grad()
def generate_kv(model, idx, n, **kw):
    """Generate n tokens WITH a KV cache.

    Phase 1 (prefill): run the whole prompt once; every layer stores its k, v.
    Phase 2 (decode):  feed only the newest token; it attends to the cached k, v of all previous tokens.

    Input:  idx (B, T0) prompt ids; kw are passed to sample_next
    Output: (B, T0 + n)  -- identical to generate_naive for greedy decoding
    """
    # One empty cache dict per layer; Attention fills in "k" and "v".
    caches = []
    for _ in model.blocks:
        caches.append({})

    # Phase 1: prefill the prompt (positions 0 .. T0-1)
    prompt_logits = model(idx, caches, pos=0)                          # (B, T0, V)
    logits = prompt_logits[:, -1]                                      # (B, V)

    # Phase 2: decode one token at a time
    out = idx
    for _ in range(n):
        next_token = sample_next(logits, **kw)                         # (B, 1)
        out = torch.cat([out, next_token], dim=1)                      # (B, T + 1)

        # The new token sits at absolute position (current length - 1); RoPE needs that position.
        new_token_pos = out.shape[1] - 1
        step_logits = model(next_token, caches, pos=new_token_pos)     # (B, 1, V)
        logits = step_logits[:, -1]                                    # (B, V)
    return out


# =====================================================================================================
# Optimisation helpers
# =====================================================================================================
def lr_at(step, max_lr, warmup, total, min_lr_ratio=0.1):
    """Learning rate schedule: linear warmup, then cosine decay down to min_lr_ratio * max_lr.

    Input:  step (int), max_lr, warmup steps, total steps
    Output: learning rate (float) for this step
    """
    # Phase 1: linear warmup from max_lr/warmup up to max_lr (avoids huge early updates while Adam stats are noisy)
    if step < warmup:
        return max_lr * (step + 1) / warmup

    # Phase 2: cosine decay. progress goes 0 -> 1 over the remaining steps.
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))                  # 1 -> 0
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * cosine


def configure_adamw(model, lr, wd=0.1):
    """AdamW with weight decay applied only to >=2D parameters (matmul weights, embeddings).

    1D parameters (norm gains, biases) are not decayed: shrinking them toward 0 hurts and gives no
    regularisation benefit.

    Output: torch.optim.AdamW with two param groups
    """
    decay_params = []
    no_decay_params = []
    for _name, param in model.named_parameters():
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


# =====================================================================================================
# Tests
# =====================================================================================================
if __name__ == "__main__":
    # ---- data: char-level tokenisation of a tiny repeated text
    text = "the quick brown fox jumps over the lazy dog. " * 50
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])

    model = GPT(len(chars))
    opt = configure_adamw(model, 3e-3)

    # ---- training loop
    T = 32          # context length
    B = 16          # micro-batch size
    steps = 300
    accum = 2       # gradient accumulation steps -> effective batch = B * accum
    first = None
    for step in range(steps):
        # set this step's learning rate on every param group
        for group in opt.param_groups:
            group["lr"] = lr_at(step, 3e-3, 30, steps)

        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            # random windows; targets are the inputs shifted left by one
            starts = torch.randint(0, len(data) - T - 1, (B,))
            x = torch.stack([data[i:i + T] for i in starts])             # (B, T)
            y = torch.stack([data[i + 1:i + T + 1] for i in starts])     # (B, T)
            logits = model(x)                                            # (B, T, V)
            # divide by accum so the summed gradients equal the gradient of the mean loss over B * accum
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # clip global grad norm to 1
        opt.step()

        if first is None:
            first = loss.item() * accum
        if step % 100 == 0:
            print(f"step {step} loss {loss.item() * accum:.3f}")
    final = loss.item() * accum
    print(f"loss {first:.3f} -> {final:.3f}")
    assert final < first * 0.3

    # ---- KV-cache generation must equal full-recompute generation (greedy)
    model.eval()
    prompt = torch.tensor([[stoi[c] for c in "the quick "]])
    a = generate_naive(model, prompt, 30, temperature=0)
    b = generate_kv(model, prompt, 30, temperature=0)

    def decode(t):
        return "".join(chars[i] for i in t[0].tolist())

    print("naive:", repr(decode(a)))
    print("kv   :", repr(decode(b)))
    assert torch.equal(a, b)

    torch.manual_seed(1)
    print("top-p:", repr(decode(generate_kv(model, prompt, 30, temperature=0.8, top_k=10, top_p=0.9))))
    print("OK")
