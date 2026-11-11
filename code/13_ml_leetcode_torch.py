"""LLM problems in LeetCode format (PyTorch): fixed signature -> implement -> hidden tests vs torch reference.

Rules (same for every problem):
  - Use only basic tensor ops (matmul, exp, log, sum, max, gather, masked_fill, view/transpose, topk, sort, ...).
  - Do NOT call the torch function that implements the thing being asked (e.g. no F.cross_entropy inside
    cross_entropy, no F.scaled_dot_product_attention / nn.MultiheadAttention inside attention).
  - Tests run in float64 and compare against the official torch implementation, autograd, or a naive loop.

Problems
  P01 softmax / log_softmax            P12 KV-cache decode loop
  P02 cross_entropy                    P13 sampling filter: temperature / top-k / top-p
  P03 causal LM loss + perplexity      P14 AdamW step
  P04 cross-entropy backward           P15 global grad-norm clipping + warmup-cosine LR
  P05 LayerNorm / RMSNorm              P16 KL: exact + k1 / k3 estimators
  P06 GELU / SiLU / SwiGLU FFN         P17 sequence log-prob + DPO + Bradley-Terry
  P07 dropout                          P18 GRPO advantages + PPO clipped loss
  P08 scaled dot-product attention     P19 MoE top-k router + load-balancing loss
  P09 RoPE / sinusoidal PE             P20 LoRA linear + merge
  P10 multi-head attention (MHA)       P21 byte-level BPE: train / encode / decode
  P11 GQA / MQA
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F


# P01 ------------------------------------------------------------------
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax of x along dimension `dim`.

    Input:  x, float tensor of any shape; dim may be any axis (including 0 or negative).
    Output: same shape as x; entries in (0, 1], summing to 1 along `dim`.
    Must be numerically stable: subtract the max along `dim` before exp, so x containing 1000.0 must not give
    inf/NaN. Must match torch.softmax(x, dim).
    """
    # Step 1: subtract the max along `dim` for numerical stability.
    #         softmax is shift-invariant, and now the largest exponent is exp(0) = 1, so nothing overflows.
    max_along_dim = x.amax(dim=dim, keepdim=True)          # same shape as x, but size 1 along `dim`
    shifted = x - max_along_dim                            # same shape as x, all values <= 0

    # Step 2: exponentiate.
    exps = torch.exp(shifted)                              # same shape as x, values in (0, 1]

    # Step 3: normalise so the values along `dim` sum to 1.
    total = exps.sum(dim=dim, keepdim=True)                # size 1 along `dim`, always >= 1
    probs = exps / total                                   # same shape as x
    return probs


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log of softmax along `dim`, computed directly (NOT log(softmax(x)), which underflows to -inf).

    log_softmax(x)_i = x_i - max(x) - log(sum_j exp(x_j - max(x))).
    Same shape as x. Must match torch.log_softmax(x, dim), including for very large / very negative logits.
    """
    # Step 1: subtract the max for numerical stability (same trick as softmax).
    max_along_dim = x.amax(dim=dim, keepdim=True)          # size 1 along `dim`
    shifted = x - max_along_dim                            # same shape as x, all values <= 0

    # Step 2: log of the normaliser, log(sum_j exp(shifted_j)).  This is the "log-sum-exp".
    #         The sum is >= 1 (the max element contributes exp(0) = 1), so the log never sees 0.
    sum_exps = torch.exp(shifted).sum(dim=dim, keepdim=True)   # size 1 along `dim`
    log_normaliser = torch.log(sum_exps)                       # size 1 along `dim`

    # Step 3: log p_i = shifted_i - log_normaliser.  We never take log of a tiny probability directly,
    #         so very negative logits give large negative numbers instead of -inf.
    log_probs = shifted - log_normaliser                   # same shape as x
    return log_probs


# P02 ------------------------------------------------------------------
def cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
    """Multi-class cross-entropy from raw logits. Must match F.cross_entropy with the same arguments.

    Args:
      logits: (N, C) float, unnormalised scores.
      target: (N,) long, class index in [0, C) or == ignore_index.
      ignore_index: rows whose target equals this contribute 0 loss and are excluded from the mean's denominator.
      label_smoothing: eps in [0, 1). Target distribution = (1 - eps) * onehot + eps / C * ones, i.e.
                       loss_i = (1 - eps) * (-log p_i[y_i]) + eps * mean_c(-log p_i[c]).
      reduction: "none" -> (N,) per-row loss (0 for ignored rows); "sum" -> scalar sum;
                 "mean" -> sum / number of NON-ignored rows (not / N).
    Careful: ignored targets (-100) are not valid gather indices.
    """
    # Step 1: log-probabilities of every class (stable, see P01).
    log_probs = log_softmax(logits, dim=-1)                # (N, C)

    # Step 2: which rows actually count?
    is_valid = target != ignore_index                      # (N,) bool
    is_valid_float = is_valid.to(logits.dtype)             # (N,) 1.0 for real rows, 0.0 for ignored rows

    # Step 3: pick out log p[y_i] for each row.
    #         Ignored rows have target -100, which is not a valid index, so we temporarily
    #         give them index 0 so gather works; their loss is zeroed out in Step 5 anyway.
    safe_target = torch.where(is_valid, target, torch.zeros_like(target))   # (N,)
    gather_index = safe_target.unsqueeze(-1)                                 # (N, 1)
    target_log_prob = log_probs.gather(dim=-1, index=gather_index)           # (N, 1)
    target_log_prob = target_log_prob.squeeze(-1)                            # (N,)
    nll_loss = -target_log_prob                                              # (N,)  -log p[y_i]

    # Step 4: label smoothing term = cross-entropy against the uniform distribution,
    #         i.e. the average of -log p[c] over all C classes.
    uniform_loss = -log_probs.mean(dim=-1)                 # (N,)

    # Step 5: mix the two targets, then zero out ignored rows.
    per_row_loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * uniform_loss   # (N,)
    per_row_loss = per_row_loss * is_valid_float                                         # (N,)

    # Step 6: reduce.
    if reduction == "none":
        return per_row_loss                                # (N,)
    if reduction == "sum":
        return per_row_loss.sum()                          # scalar

    # "mean": divide by the number of NON-ignored rows, not by N.
    num_valid_rows = is_valid_float.sum()                  # scalar
    mean_loss = per_row_loss.sum() / num_valid_rows        # scalar
    return mean_loss


# P03 ------------------------------------------------------------------
def causal_lm_loss(logits, input_ids, attention_mask, prompt_lens=None):
    """Next-token prediction loss of a causal LM, plus perplexity.

    Args:
      logits: (B, T, V) model outputs for input_ids; logits[:, t] is the prediction for token t+1.
      input_ids: (B, T) long.
      attention_mask: (B, T) long/bool, 1 = real token, 0 = empty slot after the sequence ended (not a real token).
      prompt_lens: optional (B,) long. If given, tokens at positions < prompt_lens[b] are prompt and are NOT
                   trained on (SFT: loss only on the response).
    Returns: (loss, ppl) where loss = mean NLL over all counted target tokens in the whole batch (token-weighted,
             not mean of per-sequence means) and ppl = exp(loss).
    A target token at position t (t >= 1) is counted iff attention_mask[b, t] == 1 and t >= prompt_lens[b].
    Must equal F.cross_entropy(logits[:, :-1].flatten, labels[:, 1:].flatten, ignore_index=-100) with the masked
    labels set to -100.
    """
    B, T, V = logits.shape
    IGNORE = -100

    # Step 1: labels start as a copy of the inputs (we must not modify input_ids in place).
    labels = input_ids.clone()                             # (B, T)

    # Step 2: empty slots (attention_mask == 0) are not real tokens, so never train on them.
    is_empty_slot = attention_mask == 0                    # (B, T) bool
    labels[is_empty_slot] = IGNORE

    # Step 3 (SFT): prompt tokens are not trained on either, only the response.
    if prompt_lens is not None:
        positions = torch.arange(T).unsqueeze(0)           # (1, T)   0, 1, ..., T-1
        prompt_end = prompt_lens.unsqueeze(1)              # (B, 1)
        is_prompt = positions < prompt_end                 # (B, T) bool, broadcast
        labels[is_prompt] = IGNORE

    # Step 4: shift by one. The logits at position t predict the token at t+1,
    #         so drop the last logit (nothing to predict) and the first label (nothing predicts it).
    pred_logits = logits[:, :-1, :]                        # (B, T-1, V)
    target_labels = labels[:, 1:]                          # (B, T-1)

    # Step 5: flatten batch and time so every counted token is weighted equally (token-level mean).
    flat_logits = pred_logits.reshape(-1, V)               # (B*(T-1), V)
    flat_labels = target_labels.reshape(-1)                # (B*(T-1),)
    loss = cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE)   # scalar

    # Step 6: perplexity = exp(mean NLL per token).
    perplexity = torch.exp(loss)                           # scalar
    return loss, perplexity


# P04 ------------------------------------------------------------------
def cross_entropy_backward(logits, target):
    """Gradient of mean cross-entropy w.r.t. the logits, derived by hand (no autograd).

    Args: logits (N, C) float, target (N,) long (no ignore_index here).
    Loss: L = (1/N) * sum_i -log softmax(logits_i)[target_i].
    Returns: dL/dlogits, shape (N, C) = (softmax(logits) - onehot(target)) / N.
    Must match logits.grad after F.cross_entropy(logits, target).backward().
    """
    N = logits.shape[0]

    # Step 1: predicted probabilities.
    probs = softmax(logits, dim=-1)                        # (N, C)

    # Step 2: one-hot encoding of the targets, built with basic ops.
    row_index = torch.arange(N)                            # (N,)
    one_hot = torch.zeros_like(probs)                      # (N, C)
    one_hot[row_index, target] = 1.0                       # put a 1 in each row at the target column

    # Step 3: d(-log softmax_y)/d logits = softmax - onehot, per row.
    grad_per_row = probs - one_hot                         # (N, C)

    # Step 4: the loss is a MEAN over N rows, so each row's gradient is scaled by 1/N.
    grad = grad_per_row / N                                # (N, C)
    return grad


# P05 ------------------------------------------------------------------
def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dimension.

    x: (..., D); weight, bias: (D,).
    y = (x - mean) / sqrt(var + eps) * weight + bias, mean/var over the last dim, var BIASED (divide by D).
    Must match F.layer_norm(x, (D,), weight, bias, eps).
    """
    # Step 1: mean over the feature dimension.
    mean = x.mean(dim=-1, keepdim=True)                    # (..., 1)

    # Step 2: biased variance (divide by D, not D-1) -- this is what LayerNorm uses.
    centered = x - mean                                    # (..., D)
    variance = (centered * centered).mean(dim=-1, keepdim=True)   # (..., 1)

    # Step 3: normalise to zero mean / unit variance. eps avoids division by zero.
    std = torch.sqrt(variance + eps)                       # (..., 1)
    normalized = centered / std                            # (..., D)

    # Step 4: learnable per-feature scale and shift.
    out = normalized * weight + bias                       # (..., D)
    return out


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm (Llama) over the last dimension: no mean subtraction, no bias.

    x: (..., D); weight: (D,).  y = x / sqrt(mean(x^2, last dim) + eps) * weight.
    Must match F.rms_norm(x, (D,), weight, eps).
    """
    # Step 1: root-mean-square of each vector (no mean subtraction, unlike LayerNorm).
    mean_square = (x * x).mean(dim=-1, keepdim=True)       # (..., 1)
    rms = torch.sqrt(mean_square + eps)                    # (..., 1)

    # Step 2: divide by the RMS so every vector has RMS ~1.
    normalized = x / rms                                   # (..., D)

    # Step 3: learnable per-feature scale (no bias in RMSNorm).
    out = normalized * weight                              # (..., D)
    return out


# P06 ------------------------------------------------------------------
def gelu(x, approximate="none"):
    """GELU activation, elementwise.

    approximate="none": exact,  0.5 * x * (1 + erf(x / sqrt(2))).
    approximate="tanh": GPT-2,  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Must match F.gelu(x, approximate=...).
    """
    if approximate == "tanh":
        # GPT-2 approximation: replace the Gaussian CDF with a tanh curve.
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)   # same shape as x
        gaussian_cdf = 0.5 * (1.0 + torch.tanh(inner))               # same shape as x
    else:
        # Exact: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))) is the standard-normal CDF.
        gaussian_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))   # same shape as x

    # GELU(x) = x * Phi(x): scale each input by "how likely it is to be kept".
    return x * gaussian_cdf


def silu(x):
    """SiLU / Swish, elementwise: x * sigmoid(x). Must match F.silu."""
    gate = torch.sigmoid(x)                                # same shape as x, values in (0, 1)
    return x * gate


def swiglu_ffn(x, w_gate, w_up, w_down):
    """Llama-style SwiGLU feed-forward block (no biases).

    x: (..., D); w_gate, w_up: (D, H); w_down: (H, D)   (weights are stored as (in, out), so use x @ w).
    y = (silu(x @ w_gate) * (x @ w_up)) @ w_down, shape (..., D).
    """
    # Step 1: two parallel up-projections from D to the hidden size H.
    gate_proj = x @ w_gate                                 # (..., H)
    up_proj = x @ w_up                                     # (..., H)

    # Step 2: the gate branch goes through SiLU and multiplies the other branch elementwise.
    gate = silu(gate_proj)                                 # (..., H)
    hidden = gate * up_proj                                # (..., H)

    # Step 3: project back down to the model dimension.
    out = hidden @ w_down                                  # (..., D)
    return out


# P07 ------------------------------------------------------------------
def dropout(x, p, training):
    """Inverted dropout.

    training=True: zero each element independently with probability p and scale survivors by 1 / (1 - p),
                   so E[output] == x. p == 1 -> all zeros (avoid division by zero). p == 0 -> x.
    training=False: return x unchanged (no scaling at eval time).
    Tested statistically: drop rate ~= p and mean preserved on a large tensor of ones.
    """
    # At eval time (or with p = 0) dropout does nothing.
    if not training or p == 0:
        return x

    # p = 1 drops everything; handle it separately to avoid dividing by (1 - p) = 0.
    if p == 1:
        return torch.zeros_like(x)

    # Step 1: draw a uniform random number per element; keep it if the number is >= p.
    #         P(keep) = 1 - p, P(drop) = p.
    random_values = torch.rand(x.shape, dtype=x.dtype)     # same shape as x, uniform in [0, 1)
    keep_mask = (random_values >= p).to(x.dtype)           # same shape as x, 1.0 = keep, 0.0 = drop

    # Step 2: zero dropped elements and scale survivors by 1/(1-p),
    #         so the expected value matches x and no rescaling is needed at eval time ("inverted" dropout).
    scale = 1.0 / (1.0 - p)
    out = x * keep_mask * scale                            # same shape as x
    return out


# P08 ------------------------------------------------------------------
def sdpa(q, k, v, attention_mask=None, is_causal=False):
    """Scaled dot-product attention: softmax(q k^T / sqrt(d) + mask) v.

    Args:
      q: (..., T, d) queries; k, v: (..., S, d) keys / values (leading dims, e.g. (B, H), are batch dims).
      attention_mask: optional bool tensor broadcastable to the score shape (..., T, S);
                      True = this query may attend to this key, False = it may not.
      is_causal: if True, query i may only attend to keys j <= i + (S - T). When T < S (decoding with a cache)
                 the queries are the LAST T positions, so the mask is tril(ones(T, S), diagonal=S - T).
                 If both is_causal and attention_mask are given, a key must pass both.
    Returns: (..., T, d).
    Edge case: a query row with every key masked must output all zeros (plain softmax over all -inf gives NaN).
    Must match F.scaled_dot_product_attention for the non-degenerate cases.
    """
    T = q.shape[-2]                                        # number of queries
    S = k.shape[-2]                                        # number of keys
    d = q.shape[-1]                                        # head dimension

    # Step 1: similarity score between every query and every key, scaled by 1/sqrt(d)
    #         so the dot products don't grow with d and saturate the softmax.
    k_transposed = k.transpose(-2, -1)                     # (..., d, S)
    scores = q @ k_transposed                              # (..., T, S)
    scores = scores / math.sqrt(d)                         # (..., T, S)

    # Step 2: build the "allowed" mask. Start with everything allowed.
    allowed = torch.ones(T, S, dtype=torch.bool)           # (T, S)
    if is_causal:
        # Queries are the LAST T of the S positions, so query i sits at absolute position i + (S - T)
        # and may see keys 0 .. i + (S - T). That is a lower-triangular mask shifted right by S - T.
        causal_allowed = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)   # (T, S)
        allowed = allowed & causal_allowed
    if attention_mask is not None:
        # A key must pass BOTH the causal mask and the user's attention_mask.
        allowed = allowed & attention_mask                 # broadcasts to (..., T, S)

    # Step 3: masked-out keys get -inf, so exp(-inf) = 0 weight after softmax.
    scores = scores.masked_fill(~allowed, float("-inf"))   # (..., T, S)

    # Step 4: softmax over the keys (stable: subtract the row max first).
    #         For a fully-masked row the max is -inf; replace it with 0 so we don't compute (-inf) - (-inf) = NaN.
    row_max = scores.amax(dim=-1, keepdim=True)            # (..., T, 1)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exps = torch.exp(scores - row_max)                     # (..., T, S), masked entries are exactly 0
    row_sum = exps.sum(dim=-1, keepdim=True)               # (..., T, 1), 0 for a fully-masked row

    # A fully-masked row has row_sum == 0; divide by 1 instead so its weights stay all-zero (output = 0, not NaN).
    safe_row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    weights = exps / safe_row_sum                          # (..., T, S), each valid row sums to 1

    # Step 5: weighted average of the values.
    out = weights @ v                                      # (..., T, d)
    return out


# P09 ------------------------------------------------------------------
def apply_rope(x, offset=0, base=10000.0):
    """Rotary position embedding, interleaved-pair convention.

    x: (..., T, d), d even. Token t sits at absolute position pos = offset + t (offset > 0 when decoding with a
    KV cache). For i in [0, d/2): theta_i = base^(-2i/d), angle = pos * theta_i, and the pair (x[2i], x[2i+1])
    is rotated by that angle:
        out[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
        out[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)
    Properties tested: preserves the norm of every vector; <rope(q, m), rope(k, n)> depends only on m - n;
    apply_rope(x)[:, 2:] == apply_rope(x[:, 2:], offset=2).
    """
    T = x.shape[-2]
    d = x.shape[-1]

    # Step 1: absolute position of every token (shifted by `offset` when decoding with a KV cache).
    positions = torch.arange(offset, offset + T, dtype=x.dtype)     # (T,)
    positions = positions.unsqueeze(1)                              # (T, 1)

    # Step 2: one rotation frequency per pair of dims: theta_i = base^(-2i/d).
    #         Early pairs rotate fast, late pairs rotate slowly.
    pair_index_times_2 = torch.arange(0, d, 2, dtype=x.dtype)       # (d/2,)  0, 2, 4, ...
    theta = base ** (-pair_index_times_2 / d)                       # (d/2,)

    # Step 3: rotation angle for every (position, pair).
    angles = positions * theta                                      # (T, d/2), broadcast
    cos = torch.cos(angles)                                         # (T, d/2)
    sin = torch.sin(angles)                                         # (T, d/2)

    # Step 4: split x into the even and odd members of each pair.
    x_even = x[..., 0::2]                                           # (..., T, d/2)  x[2i]
    x_odd = x[..., 1::2]                                            # (..., T, d/2)  x[2i+1]

    # Step 5: apply the 2-D rotation [[cos, -sin], [sin, cos]] to each pair.
    rotated_even = x_even * cos - x_odd * sin                       # (..., T, d/2)
    rotated_odd = x_even * sin + x_odd * cos                        # (..., T, d/2)

    # Step 6: interleave the rotated halves back into their original slots.
    out = torch.empty_like(x)                                       # (..., T, d)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def sinusoidal_pe(T, d, base=10000.0):
    """Original Transformer absolute positional encoding, returns (T, d) float64.

    pe[pos, 2i] = sin(pos / base^(2i/d)), pe[pos, 2i+1] = cos(pos / base^(2i/d)). Row 0 = [0, 1, 0, 1, ...].
    """
    # Step 1: positions 0 .. T-1 as a column.
    positions = torch.arange(T, dtype=torch.float64).unsqueeze(1)   # (T, 1)

    # Step 2: one frequency per pair of dims: 1 / base^(2i/d).
    pair_index_times_2 = torch.arange(0, d, 2, dtype=torch.float64) # (d/2,)
    inv_freq = base ** (-pair_index_times_2 / d)                    # (d/2,)

    # Step 3: angle for every (position, pair).
    angles = positions * inv_freq                                   # (T, d/2)

    # Step 4: even dims get sin, odd dims get cos.
    pe = torch.zeros(T, d, dtype=torch.float64)                     # (T, d)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


# P10 ------------------------------------------------------------------
def multi_head_attention(x, w_qkv, b_qkv, w_o, b_o, n_heads, attention_mask=None, causal=False):
    """Standard multi-head self-attention, matching nn.MultiheadAttention(batch_first=True) weight layout.

    Args:
      x: (B, T, D) input. D must be divisible by n_heads; head dim Dh = D // n_heads.
      w_qkv: (3D, D) packed input projection = [W_q; W_k; W_v] stacked along dim 0 (PyTorch layout, out x in),
             so q = x @ W_q^T + b_q, etc.  b_qkv: (3D,) packed biases.
      w_o: (D, D) output projection, b_o: (D,)  ->  out = concat_heads @ w_o^T + b_o.
      n_heads: number of heads H.
      attention_mask: optional bool tensor of shape (B, T, T) (or broadcastable to it, e.g. (T, T));
                      attention_mask[b, i, j] = True means query i may attend to key j. Same for every head.
      causal: if True, position i attends only to positions <= i (combined with attention_mask: must pass both).
    Steps: project -> split into heads (B, H, T, Dh) -> scaled dot-product attention per head (scale 1/sqrt(Dh))
           -> merge heads back to (B, T, D) (transpose, then reshape) -> output projection.
    Returns: (B, T, D). No dropout. You may reuse your own sdpa from P08, not F.scaled_dot_product_attention.
    """
    B, T, D = x.shape
    H = n_heads
    Dh = D // H

    # Step 1: one packed projection for q, k, v (PyTorch layout is (out, in), hence the transpose).
    qkv = x @ w_qkv.T + b_qkv                              # (B, T, 3D)

    # Step 2: split the packed result into q, k, v. Rows of w_qkv are [W_q; W_k; W_v] in that order.
    q = qkv[..., 0 * D:1 * D]                              # (B, T, D)
    k = qkv[..., 1 * D:2 * D]                              # (B, T, D)
    v = qkv[..., 2 * D:3 * D]                              # (B, T, D)

    # Step 3: split the feature dim into H heads, then move heads in front of time
    #         so each head is an independent attention problem.
    q = q.reshape(B, T, H, Dh).transpose(1, 2)             # (B, H, T, Dh)
    k = k.reshape(B, T, H, Dh).transpose(1, 2)             # (B, H, T, Dh)
    v = v.reshape(B, T, H, Dh).transpose(1, 2)             # (B, H, T, Dh)

    # Step 4: the attention mask is the same for every head, so add a head axis of size 1 to broadcast.
    #         (B, T, T) -> (B, 1, T, T). A 2-D (T, T) mask already broadcasts against (B, H, T, T).
    head_mask = attention_mask
    if attention_mask is not None and attention_mask.dim() == 3:
        head_mask = attention_mask.unsqueeze(1)            # (B, 1, T, T)

    # Step 5: scaled dot-product attention per head (sdpa combines the causal mask with head_mask).
    head_outputs = sdpa(q, k, v, attention_mask=head_mask, is_causal=causal)   # (B, H, T, Dh)

    # Step 6: merge heads: move time back in front of heads, then concatenate the heads along features.
    merged = head_outputs.transpose(1, 2)                  # (B, T, H, Dh)
    merged = merged.reshape(B, T, D)                       # (B, T, D)

    # Step 7: output projection.
    out = merged @ w_o.T + b_o                             # (B, T, D)
    return out


# P11 ------------------------------------------------------------------
def gqa_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, causal=True):
    """Grouped-query attention (covers MHA when n_kv_heads == n_heads and MQA when n_kv_heads == 1).

    Args:
      x: (B, T, D); Dh = D // n_heads; group size G = n_heads // n_kv_heads.
      wq: (D, D); wk, wv: (D, n_kv_heads * Dh); wo: (D, D). Weights are (in, out): q = x @ wq. No biases.
      causal: causal self-attention if True.
    Query head h uses KV head h // G (consecutive query heads share a KV head, Llama convention), i.e. K/V are
    expanded with repeat_interleave(G, dim=head), not repeat/tile.
    Returns: (B, T, D) = merged heads @ wo. Must match F.scaled_dot_product_attention(..., enable_gqa=True).
    """
    B, T, D = x.shape
    H = n_heads
    H_kv = n_kv_heads
    Dh = D // H
    G = H // H_kv                                          # query heads per KV head

    # Step 1: project. Queries get H heads; keys/values only H_kv heads (that's the memory saving).
    q = x @ wq                                             # (B, T, H * Dh)
    k = x @ wk                                             # (B, T, H_kv * Dh)
    v = x @ wv                                             # (B, T, H_kv * Dh)

    # Step 2: split into heads and put the head axis before time.
    q = q.reshape(B, T, H, Dh).transpose(1, 2)             # (B, H, T, Dh)
    k = k.reshape(B, T, H_kv, Dh).transpose(1, 2)          # (B, H_kv, T, Dh)
    v = v.reshape(B, T, H_kv, Dh).transpose(1, 2)          # (B, H_kv, T, Dh)

    # Step 3: share each KV head with G consecutive query heads.
    #         repeat_interleave gives [kv0, kv0, kv1, kv1, ...], so query head h uses KV head h // G.
    k = k.repeat_interleave(G, dim=1)                      # (B, H, T, Dh)
    v = v.repeat_interleave(G, dim=1)                      # (B, H, T, Dh)

    # Step 4: ordinary attention per head.
    head_outputs = sdpa(q, k, v, is_causal=causal)         # (B, H, T, Dh)

    # Step 5: merge heads and apply the output projection.
    merged = head_outputs.transpose(1, 2)                  # (B, T, H, Dh)
    merged = merged.reshape(B, T, D)                       # (B, T, D)
    out = merged @ wo                                      # (B, T, D)
    return out


# P12 ------------------------------------------------------------------
def decode_with_kv_cache(x, wq, wk, wv, n_heads):
    """Single-layer causal self-attention computed one token at a time with a growing KV cache.

    Args: x (B, T, D); wq, wk, wv: (D, D), (in, out) layout, no biases, no output projection.
    For t = 0 .. T-1: project ONLY token t to q, k, v (B, H, 1, Dh); apply RoPE (P09) to q and k with
    offset=t (absolute position) BEFORE appending k to the cache; append k, v to the cache along the time axis;
    attend the single query to all cached keys (no mask needed: the cache only holds positions <= t).
    Returns: (B, T, D) concatenated per-step outputs with heads merged. Must equal the full causal forward
    (RoPE on all positions + causal SDPA) to 1e-6.
    """
    B, T, D = x.shape
    H = n_heads
    Dh = D // H

    # The cache starts empty (time length 0) and grows by one position per step.
    k_cache = torch.zeros(B, H, 0, Dh, dtype=x.dtype)      # (B, H, 0, Dh)
    v_cache = torch.zeros(B, H, 0, Dh, dtype=x.dtype)      # (B, H, 0, Dh)
    step_outputs = []

    for t in range(T):
        # Step 1: take only the current token (keep the time axis with size 1).
        x_t = x[:, t:t + 1, :]                             # (B, 1, D)

        # Step 2: project just this token into q, k, v and split into heads.
        q_t = (x_t @ wq).reshape(B, 1, H, Dh).transpose(1, 2)   # (B, H, 1, Dh)
        k_t = (x_t @ wk).reshape(B, 1, H, Dh).transpose(1, 2)   # (B, H, 1, Dh)
        v_t = (x_t @ wv).reshape(B, 1, H, Dh).transpose(1, 2)   # (B, H, 1, Dh)

        # Step 3: RoPE with the ABSOLUTE position t. Keys are rotated BEFORE caching,
        #         so cached keys never need to be touched again.
        q_t = apply_rope(q_t, offset=t)                    # (B, H, 1, Dh)
        k_t = apply_rope(k_t, offset=t)                    # (B, H, 1, Dh)

        # Step 4: append this step's key and value to the cache along the time axis.
        k_cache = torch.cat([k_cache, k_t], dim=2)         # (B, H, t+1, Dh)
        v_cache = torch.cat([v_cache, v_t], dim=2)         # (B, H, t+1, Dh)

        # Step 5: the single query attends to everything in the cache.
        #         No causal mask needed: the cache only contains positions 0..t.
        out_t = sdpa(q_t, k_cache, v_cache)                # (B, H, 1, Dh)
        step_outputs.append(out_t)

    # Step 6: stitch the per-step outputs together along time, then merge heads.
    all_outputs = torch.cat(step_outputs, dim=2)           # (B, H, T, Dh)
    all_outputs = all_outputs.transpose(1, 2)              # (B, T, H, Dh)
    out = all_outputs.reshape(B, T, D)                     # (B, T, D)
    return out


# P13 ------------------------------------------------------------------
def filter_logits(logits, temperature=1.0, k=0, p=1.0):
    """Prepare next-token logits for sampling. Returns a NEW (B, V) tensor; removed tokens set to -inf.

    Order: temperature, then top-k, then top-p.
      temperature: divide logits by it. temperature == 0 means greedy: keep only the argmax (value 0), rest -inf.
      top-k (k > 0): keep the k largest logits per row.
      top-p (p < 1): sort descending; keep the smallest prefix whose cumulative probability is >= p
                     (a token is kept iff the probability mass strictly before it is < p). Always keep the top-1.
    Example probs [0.5, 0.3, 0.15, 0.05]: p=0.75 keeps 2 tokens, p=0.85 keeps 3, p=0.1 keeps 1.
    Do not modify the input in place.
    """
    NEG_INF = float("-inf")

    # Greedy decoding: only the argmax survives (with logit 0), everything else is -inf.
    if temperature == 0:
        best_index = logits.argmax(dim=-1, keepdim=True)               # (B, 1)
        out = torch.full_like(logits, NEG_INF)                         # (B, V)
        out = out.scatter(dim=-1, index=best_index, value=0.0)         # (B, V)
        return out

    # Step 1: temperature. Division creates a new tensor, so the input is never modified.
    #         T > 1 flattens the distribution, T < 1 sharpens it.
    out = logits / temperature                                         # (B, V)

    # Step 2: top-k. Find the k-th largest value per row and drop everything strictly below it.
    if k > 0:
        top_k_values = out.topk(k, dim=-1).values                      # (B, k), sorted descending
        kth_largest = top_k_values[:, -1:]                             # (B, 1)
        below_kth = out < kth_largest                                  # (B, V) bool
        out = out.masked_fill(below_kth, NEG_INF)                      # (B, V)

    # Step 3: top-p (nucleus).
    if p < 1.0:
        # 3a: sort each row from most to least likely, remembering where each token came from.
        sorted_logits, sorted_index = out.sort(dim=-1, descending=True)   # (B, V), (B, V)
        sorted_probs = softmax(sorted_logits, dim=-1)                     # (B, V)

        # 3b: probability mass strictly BEFORE each token (exclusive cumulative sum).
        cumulative_probs = sorted_probs.cumsum(dim=-1)                    # (B, V) inclusive
        mass_before = cumulative_probs - sorted_probs                     # (B, V) exclusive

        # 3c: a token is removed once the tokens before it already cover >= p of the mass.
        remove_sorted = mass_before >= p                                  # (B, V) bool
        remove_sorted[:, 0] = False                                       # always keep the top-1 token

        # 3d: apply the removal in sorted order, then scatter back to the original vocab order.
        filtered_sorted = sorted_logits.masked_fill(remove_sorted, NEG_INF)   # (B, V)
        out = torch.empty_like(out)                                           # (B, V)
        out = out.scatter(dim=-1, index=sorted_index, src=filtered_sorted)    # (B, V)

    return out


# P14 ------------------------------------------------------------------
def adamw_step(p, g, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
    """One AdamW update, IN PLACE on p (param), m (1st moment), v (2nd moment). Returns nothing.

    g: gradient (same shape as p). t: 1-based step count (first call t=1).
      p <- p * (1 - lr * wd)                      # decoupled weight decay, NOT added to g
      m <- b1 * m + (1 - b1) * g
      v <- b2 * v + (1 - b2) * g^2
      m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)
      p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    Must match torch.optim.AdamW over several steps.
    """
    beta1, beta2 = betas

    # Step 1: decoupled weight decay -- shrink the weights directly instead of adding wd * p to the gradient.
    #         (This is the difference between AdamW and Adam + L2.)
    p.mul_(1.0 - lr * wd)

    # Step 2: update the running mean of the gradient (1st moment), in place.
    m.mul_(beta1)
    m.add_((1.0 - beta1) * g)

    # Step 3: update the running mean of the squared gradient (2nd moment), in place.
    v.mul_(beta2)
    v.add_((1.0 - beta2) * g * g)

    # Step 4: bias correction. m and v start at 0, so early on they underestimate the true moments;
    #         dividing by (1 - beta^t) fixes that (the correction fades to 1 as t grows).
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    # Step 5: parameter update, normalised per element by the gradient's RMS. eps avoids division by 0.
    update = lr * m_hat / (torch.sqrt(v_hat) + eps)
    p.sub_(update)


# P15 ------------------------------------------------------------------
def clip_grad_norm(grads, max_norm, eps=1e-6):
    """Global-norm gradient clipping, like torch.nn.utils.clip_grad_norm_.

    grads: list of tensors of different shapes, modified IN PLACE.
    total = sqrt(sum over ALL tensors of sum(g^2))   (one norm across all params, not per tensor).
    If total > max_norm, scale every gradient by max_norm / (total + eps); otherwise leave unchanged.
    Returns: total norm BEFORE clipping (0-dim tensor).
    """
    # Step 1: ONE global L2 norm across every gradient tensor (as if they were concatenated into one vector).
    sum_of_squares = torch.zeros((), dtype=torch.float64)  # scalar accumulator
    for g in grads:
        g_double = g.double()
        sum_of_squares = sum_of_squares + (g_double * g_double).sum()
    total_norm = torch.sqrt(sum_of_squares)                # scalar

    # Step 2: scale factor. If the norm is already small enough, the factor is 1 (no change).
    clip_coef = max_norm / (total_norm.item() + eps)
    if clip_coef > 1.0:
        clip_coef = 1.0

    # Step 3: rescale every gradient IN PLACE by the same factor, so the direction is preserved.
    for g in grads:
        g.mul_(clip_coef)

    # Return the norm measured BEFORE clipping (useful for logging).
    return total_norm


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """Learning rate at 0-based `step` for linear warmup + cosine decay.

    step < warmup:  max_lr * (step + 1) / warmup        (so step warmup-1 reaches max_lr)
    otherwise:      progress = (step - warmup) / (total - warmup), clipped to [0, 1];
                    min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    So lr(warmup) = max_lr, lr at the midpoint of decay = (max_lr + min_lr) / 2, lr(>= total) = min_lr.
    """
    # Phase 1: linear warmup from max_lr/warmup up to max_lr.
    if step < warmup:
        warmup_fraction = (step + 1) / warmup
        return max_lr * warmup_fraction

    # Phase 2: cosine decay from max_lr down to min_lr.
    decay_steps = max(1, total - warmup)                   # guard against division by zero
    progress = (step - warmup) / decay_steps               # 0 at the end of warmup, 1 at `total`
    progress = min(1.0, progress)                          # stay at min_lr after `total`

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))   # goes smoothly from 1 to 0
    lr = min_lr + (max_lr - min_lr) * cosine_factor
    return lr


# P16 ------------------------------------------------------------------
def kl_from_logits(p_logits, q_logits):
    """Exact KL(P || Q) between two categorical distributions given as logits.

    p_logits, q_logits: (N, C). KL_n = sum_c P[n,c] * (log P[n,c] - log Q[n,c]); return the mean over N (scalar).
    Compute with log_softmax for stability. Note F.kl_div(input=log Q, target=log P) is the torch equivalent.
    """
    # Step 1: log-probabilities of both distributions (stable).
    log_p = log_softmax(p_logits, dim=-1)                  # (N, C)
    log_q = log_softmax(q_logits, dim=-1)                  # (N, C)

    # Step 2: KL = E_P[log P - log Q] = sum_c P * (log P - log Q).
    p = torch.exp(log_p)                                   # (N, C)
    log_ratio = log_p - log_q                              # (N, C)
    kl_per_row = (p * log_ratio).sum(dim=-1)               # (N,)

    # Step 3: average over the batch.
    return kl_per_row.mean()                               # scalar


def kl_estimators(logp, logq):
    """Monte-Carlo estimators of KL(P || Q) from samples x ~ P (as used for the KL penalty in RLHF / GRPO).

    logp, logq: tensors of log P(x_i) and log Q(x_i) for samples x_i drawn from P (same shape).
    Return (k1, k3), per-sample estimates with the same shape:
      k1 = log p - log q                     (unbiased, high variance, can be negative)
      k3 = r - 1 - log r,  r = q / p          (unbiased, always >= 0, lower variance)
    The test checks that both means converge to the exact KL and that k3 >= 0.
    """
    # log r = log(q / p), computed in log space so we never divide tiny probabilities.
    log_r = logq - logp                                    # same shape as inputs
    r = torch.exp(log_r)                                   # same shape

    # k1: the naive estimator log p - log q = -log r.
    k1 = -log_r

    # k3: add the zero-mean control variate (r - 1) (E_P[r] = 1). Since r - 1 >= log r, k3 is always >= 0.
    k3 = (r - 1.0) - log_r
    return k1, k3


# P17 ------------------------------------------------------------------
def sequence_logprob(logits, input_ids, loss_mask):
    """Total log-probability the model assigns to the selected tokens of each sequence (used by DPO / RL).

    logits: (B, T, V) for input_ids (B, T); loss_mask: (B, T), 1 on tokens that should be scored (the response).
    Token t (t >= 1) is predicted by logits[:, t-1]. Return (B,) = sum over t>=1 of loss_mask[b,t] * log p(ids[b,t]).
    SUM over tokens, not mean.
    """
    # Step 1: shift by one -- logits at position t-1 predict the token at position t.
    pred_logits = logits[:, :-1, :]                        # (B, T-1, V)
    target_ids = input_ids[:, 1:]                          # (B, T-1)
    target_mask = loss_mask[:, 1:]                         # (B, T-1)

    # Step 2: log-probabilities over the vocabulary.
    log_probs = log_softmax(pred_logits, dim=-1)           # (B, T-1, V)

    # Step 3: pick the log-prob of the token that actually appears next.
    gather_index = target_ids.unsqueeze(-1)                # (B, T-1, 1)
    token_log_probs = log_probs.gather(dim=-1, index=gather_index)   # (B, T-1, 1)
    token_log_probs = token_log_probs.squeeze(-1)                    # (B, T-1)

    # Step 4: keep only the scored tokens (the response) and SUM them per sequence.
    masked_log_probs = token_log_probs * target_mask       # (B, T-1)
    seq_log_prob = masked_log_probs.sum(dim=-1)            # (B,)
    return seq_log_prob


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimization loss.

    Inputs: (B,) sequence log-probs (from sequence_logprob) of the chosen / rejected responses under the policy
    (pi_*) and the frozen reference model (ref_*).
      z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
      loss = mean(-log sigmoid(z)) = mean(softplus(-z))
    Must be stable for |z| in the hundreds (do not compute log(sigmoid(z)) naively).
    Sanity: policy == reference -> loss = log 2.
    """
    # Step 1: implicit rewards = how much the policy moved away from the reference on each response.
    chosen_log_ratio = pi_chosen - ref_chosen              # (B,)
    rejected_log_ratio = pi_rejected - ref_rejected        # (B,)

    # Step 2: margin between chosen and rejected, scaled by beta.
    z = beta * (chosen_log_ratio - rejected_log_ratio)     # (B,)

    # Step 3: -log sigmoid(z) = softplus(-z) = log(1 + exp(-z)).
    #         Naive exp(-z) overflows for z = -800, so rewrite it as max(-z, 0) + log(1 + exp(-|z|)):
    #         the exp argument is now always <= 0 and can never overflow.
    positive_part = torch.clamp(-z, min=0.0)               # (B,)  max(-z, 0)
    correction = torch.log1p(torch.exp(-torch.abs(z)))     # (B,)  in (0, log 2]
    loss_per_pair = positive_part + correction             # (B,)
    return loss_per_pair.mean()                            # scalar


def bradley_terry_loss(r_chosen, r_rejected):
    """Reward-model pairwise loss: mean(-log sigmoid(r_chosen - r_rejected)). r_*: (B,) scalar rewards.
    Must be finite even when r_rejected - r_chosen = 1600."""
    # The model should give the chosen response a higher reward: P(chosen > rejected) = sigmoid(margin).
    margin = r_chosen - r_rejected                         # (B,)

    # -log sigmoid(margin) = softplus(-margin), written as max(-margin, 0) + log(1 + exp(-|margin|))
    # so the exp argument is always <= 0 and a margin of -1600 stays finite (same trick as dpo_loss).
    positive_part = torch.clamp(-margin, min=0.0)              # (B,)
    correction = torch.log1p(torch.exp(-torch.abs(margin)))    # (B,)
    loss_per_pair = positive_part + correction                 # (B,)
    return loss_per_pair.mean()                            # scalar


# P18 ------------------------------------------------------------------
def grpo_advantages(rewards, group_size, eps=1e-6):
    """GRPO advantage: normalise rewards within each group of samples from the same prompt.

    rewards: (num_prompts * group_size,) laid out group by group ([p0s0, p0s1, ..., p1s0, ...]).
    A = (r - mean_group) / (std_group + eps), std = unbiased (torch default). Returns the same shape.
    If all rewards in a group are equal, the advantages for that group are 0 (no learning signal).
    """
    # Step 1: one row per prompt, one column per sample of that prompt.
    grouped = rewards.reshape(-1, group_size)              # (P, G)

    # Step 2: per-group statistics (unbiased std, the torch default).
    group_mean = grouped.mean(dim=-1, keepdim=True)        # (P, 1)
    group_std = grouped.std(dim=-1, keepdim=True)          # (P, 1)

    # Step 3: z-score within each group. eps keeps it finite when all rewards are equal (std = 0 -> A = 0).
    advantages = (grouped - group_mean) / (group_std + eps)   # (P, G)

    # Step 4: back to the original flat layout.
    return advantages.reshape(-1)                          # (P * G,)


def ppo_clip_loss(logp_new, logp_old, advantages, mask, clip=0.2):
    """PPO clipped surrogate loss at token level (as used in GRPO / RLHF).

    logp_new: (B, T) log-probs of the sampled tokens under the current policy (requires grad).
    logp_old: (B, T) log-probs under the policy that generated the samples (constant).
    advantages: (B,) one per sequence (broadcast to all its tokens) or (B, T) per token.
    mask: (B, T), 1 on response tokens.
      ratio = exp(logp_new - logp_old)
      obj   = min(ratio * A, clip(ratio, 1 - clip, 1 + clip) * A)
      loss  = -sum(obj * mask) / sum(mask)        (mean over all response tokens in the batch)
    Tested: ratio == 1 gives -mean(A); a token with A > 0 and ratio > 1 + clip gets zero gradient.
    """
    # Step 1: a per-sequence advantage (B,) is shared by all tokens of that sequence -> (B, 1) broadcasts over T.
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)               # (B, 1)

    # Step 2: importance ratio pi_new / pi_old, computed from log-probs.
    ratio = torch.exp(logp_new - logp_old)                 # (B, T)

    # Step 3: the two candidate objectives: unclipped, and with the ratio clipped to [1 - clip, 1 + clip].
    unclipped_obj = ratio * advantages                     # (B, T)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)   # (B, T)
    clipped_obj = clipped_ratio * advantages               # (B, T)

    # Step 4: take the pessimistic (smaller) one -- this stops the policy from moving too far in one update.
    objective = torch.minimum(unclipped_obj, clipped_obj)  # (B, T)

    # Step 5: average over response tokens only, and negate (we maximise the objective = minimise the loss).
    masked_objective = objective * mask                    # (B, T)
    num_tokens = mask.sum()                                # scalar
    loss = -masked_objective.sum() / num_tokens            # scalar
    return loss


# P19 ------------------------------------------------------------------
def moe_forward(x, router_w, experts, k=2):
    """Top-k mixture-of-experts layer with a Switch-Transformer load-balancing auxiliary loss.

    Args:
      x: (N, D) tokens (batch and sequence already flattened).
      router_w: (D, E) router weights; router logits = x @ router_w.
      experts: list of E callables, each maps (n, D) -> (n, D).
      k: experts per token.
    Routing: probs = softmax(router logits) over ALL E experts; pick the top-k experts per token; gates = their
    probs renormalised to sum to 1 per token. y[n] = sum_j gate[n, j] * experts[idx[n, j]](x[n]).
    Run each expert once on the batch of tokens routed to it (no per-token Python loop).
    Aux loss: f_i = (number of top-k slots assigned to expert i) / (N * k); P_i = mean over tokens of probs[:, i];
              aux = E * sum_i f_i * P_i  (= 1 for a perfectly uniform router).
    Returns: (y of shape (N, D), aux scalar).
    """
    N = x.shape[0]
    E = router_w.shape[1]

    # Step 1: router probabilities over ALL experts.
    router_logits = x @ router_w                           # (N, E)
    router_probs = softmax(router_logits, dim=-1)          # (N, E)

    # Step 2: choose the top-k experts per token.
    top_probs, top_experts = router_probs.topk(k, dim=-1)  # (N, k), (N, k)

    # Step 3: renormalise the chosen probabilities so each token's k gates sum to 1.
    gate_sum = top_probs.sum(dim=-1, keepdim=True)         # (N, 1)
    gates = top_probs / gate_sum                           # (N, k)

    # Step 4: dispatch. Loop over EXPERTS (not tokens): gather every token routed to expert e,
    #         run the expert once on that batch, and add the gated outputs back to those tokens.
    y = torch.zeros_like(x)                                # (N, D)
    for e in range(E):
        routed_to_e = top_experts == e                     # (N, k) bool
        token_index, slot_index = torch.nonzero(routed_to_e, as_tuple=True)   # (n_e,), (n_e,)
        if token_index.numel() == 0:
            continue                                       # no tokens picked this expert

        expert_input = x[token_index]                      # (n_e, D)
        expert_output = experts[e](expert_input)           # (n_e, D)
        expert_gate = gates[token_index, slot_index]       # (n_e,)
        weighted_output = expert_gate.unsqueeze(1) * expert_output   # (n_e, D)

        # index_add_ accumulates; a token never picks the same expert twice, so no clashes within one expert.
        y.index_add_(0, token_index, weighted_output)      # (N, D)

    # Step 5: load-balancing loss.
    #   f_i = fraction of all N*k routing slots that went to expert i  (hard counts, no gradient)
    #   P_i = average router probability for expert i                  (soft, carries the gradient)
    slots_per_expert = torch.bincount(top_experts.reshape(-1), minlength=E)   # (E,)
    fraction_routed = slots_per_expert.to(x.dtype) / (N * k)                  # (E,)
    mean_router_prob = router_probs.mean(dim=0)                               # (E,)
    aux_loss = E * (fraction_routed * mean_router_prob).sum()                 # scalar, = 1 when uniform

    return y, aux_loss


# P20 ------------------------------------------------------------------
def lora_linear(x, W, A, B, alpha, r):
    """Linear layer with a LoRA adapter (no bias).

    x: (..., in); W: (out, in) frozen base weight (PyTorch layout); A: (r, in); B: (out, r).
    y = x @ W^T + (alpha / r) * (x @ A^T) @ B^T   — compute the low-rank path as two thin matmuls, do not form B @ A.
    With B initialised to zeros, the output equals the base layer exactly.
    """
    # Step 1: frozen base layer.
    base_out = x @ W.T                                     # (..., out)

    # Step 2: low-rank path as two thin matmuls: down-project to rank r, then up-project.
    #         Never build the full (out, in) matrix B @ A -- that would defeat the point of LoRA.
    down = x @ A.T                                         # (..., r)
    up = down @ B.T                                        # (..., out)

    # Step 3: scale the adapter by alpha / r and add it to the base output.
    scaling = alpha / r
    out = base_out + scaling * up                          # (..., out)
    return out


def lora_merge(W, A, B, alpha, r):
    """Fold the adapter into the base weight for inference: W' = W + (alpha / r) * B @ A, shape (out, in).
    x @ W'^T must equal lora_linear(x, W, A, B, alpha, r)."""
    # For inference we can afford the full delta once, so the adapter costs nothing per token.
    scaling = alpha / r
    delta_W = B @ A                                        # (out, in)
    merged_W = W + scaling * delta_W                       # (out, in)
    return merged_W


# P21 ------------------------------------------------------------------
def bpe_train(text, n_merges):
    """Train a byte-level BPE tokenizer (GPT-2 / minbpe style, no regex pre-split, no special tokens).

    Start from the UTF-8 bytes of `text` (ids 0..255). Repeat n_merges times (stop early if < 2 tokens left):
    count all adjacent pairs, take the most frequent (ties -> the pair that appears first in the sequence),
    give it the next new id (256, 257, ...), and replace every non-overlapping occurrence left to right.
    Returns: list of merges in training order: [((id_a, id_b), new_id), ...].
    """
    # Start from raw bytes: every token id is in 0..255.
    ids = list(text.encode("utf-8"))
    merges = []

    for merge_number in range(n_merges):
        # Need at least two tokens to form a pair.
        if len(ids) < 2:
            break

        # Step 1: count every adjacent pair. Counter keeps first-seen order, which we use for tie-breaking.
        pair_counts = Counter()
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pair_counts[pair] += 1

        # Step 2: find the most frequent pair. Using strict ">" keeps the EARLIEST pair on ties.
        best_pair = None
        best_count = 0
        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count

        # Step 3: give it the next unused id and record the merge.
        new_id = 256 + merge_number
        merges.append((best_pair, new_id))

        # Step 4: replace every occurrence of the pair in the sequence.
        ids = _merge(ids, best_pair, new_id)

    return merges


def _merge(ids, pair, new):
    """Helper: replace each non-overlapping occurrence of `pair` in list `ids` (scanning left to right) by `new`."""
    merged_ids = []
    i = 0
    while i < len(ids):
        is_last_token = i == len(ids) - 1
        if not is_last_token and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            # Found the pair: emit the new id and skip over both tokens (so occurrences can't overlap).
            merged_ids.append(new)
            i += 2
        else:
            # No match here: copy the token unchanged.
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


def bpe_encode(text, merges):
    """Encode a string: UTF-8 bytes, then apply the merges in training order. Returns list of ints.
    Must work for text never seen in training (falls back to bytes), including multi-byte characters like 你好."""
    # Step 1: start from raw bytes, so ANY string (even unseen characters) is representable.
    ids = list(text.encode("utf-8"))

    # Step 2: replay the merges in the same order they were learned.
    #         Order matters: later merges may build on ids created by earlier ones.
    for pair, new_id in merges:
        ids = _merge(ids, pair, new_id)
    return ids


def bpe_decode(ids, merges):
    """Inverse of bpe_encode: build id -> bytes (0..255 are single bytes, each merge = concat of its pair),
    join the bytes, decode as UTF-8. decode(encode(s)) == s for any string s."""
    # Step 1: base vocabulary -- ids 0..255 are single bytes.
    id_to_bytes = {}
    for byte_value in range(256):
        id_to_bytes[byte_value] = bytes([byte_value])

    # Step 2: every merged id is the concatenation of its two parts.
    #         Processing in training order guarantees both parts are already in the table.
    for (left_id, right_id), new_id in merges:
        id_to_bytes[new_id] = id_to_bytes[left_id] + id_to_bytes[right_id]

    # Step 3: look up each token's bytes, join them, and decode as UTF-8.
    byte_chunks = []
    for token_id in ids:
        byte_chunks.append(id_to_bytes[token_id])
    all_bytes = b"".join(byte_chunks)
    return all_bytes.decode("utf-8", errors="replace")


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
    print("P03 causal LM loss ok (shift + empty-slot mask + prompt mask)")

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
    close(sdpa(q, k, v, attention_mask=m), F.scaled_dot_product_attention(q, k, v, attn_mask=m))
    m[1] = False
    assert sdpa(q, k, v, attention_mask=m)[..., 1, :].abs().max() == 0
    print("P08 sdpa ok (causal T!=S, bool mask, fully masked row -> 0)")

    xr = torch.randn(2, 6, 8, dtype=dt)
    close(apply_rope(xr).norm(dim=-1), xr.norm(dim=-1))
    qq, kk = torch.randn(1, 1, 8, dtype=dt), torch.randn(1, 1, 8, dtype=dt)
    close((apply_rope(qq, 3) * apply_rope(kk, 1)).sum(), (apply_rope(qq, 10) * apply_rope(kk, 8)).sum())
    close(apply_rope(xr, 0)[:, 2:], apply_rope(xr[:, 2:], 2))
    assert torch.allclose(sinusoidal_pe(6, 8)[0], torch.tensor([0., 1.] * 4, dtype=dt))
    print("P09 RoPE ok (norm kept, relative position, offset) / sinusoidal ok")

    # P10 MHA (weights from nn.MultiheadAttention) vs F.scaled_dot_product_attention reference
    D, H = 32, 8
    Dh = D // H
    mha_ref = torch.nn.MultiheadAttention(D, H, batch_first=True, dtype=dt)
    w_in, b_in = mha_ref.in_proj_weight.detach(), mha_ref.in_proj_bias.detach()
    w_out, b_out = mha_ref.out_proj.weight.detach(), mha_ref.out_proj.bias.detach()
    xm_ = torch.randn(3, 7, D, dtype=dt)
    am3 = torch.ones(3, 7, 7, dtype=torch.bool)                    # True = query may attend to key
    am3[1, :, 5:] = False; am3[2, :, 2:] = False                   # seq 1 has 5 real tokens, seq 2 has 2
    causal_ok = torch.ones(7, 7, dtype=torch.bool).tril()
    def mha_reference(mask):
        qr, kr, vr = (xm_ @ w_in.T + b_in).split(D, -1)
        qr, kr, vr = (t.view(3, 7, H, Dh).transpose(1, 2) for t in (qr, kr, vr))
        o = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=None if mask is None else mask.unsqueeze(1))
        return o.transpose(1, 2).reshape(3, 7, D) @ w_out.T + b_out
    args = (w_in, b_in, w_out, b_out, H)
    for am_, causal in [(None, False), (am3, False), (None, True), (am3, True)]:
        if causal:
            ref_mask = causal_ok.expand(3, 7, 7) if am_ is None else am_ & causal_ok
        else:
            ref_mask = am_
        close(multi_head_attention(xm_, *args, attention_mask=am_, causal=causal), mha_reference(ref_mask))
    print("P10 MHA ok (vs SDPA reference: plain / attention_mask / causal / both)")

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
    print("P11 MHA/GQA/MQA ok (vs SDPA enable_gqa)")

    wq, wk, wv = (torch.randn(D, D, dtype=dt) for _ in range(3))
    Bq, Tq = 2, 9
    xs = torch.randn(Bq, Tq, D, dtype=dt)
    qf = apply_rope((xs @ wq).view(Bq, Tq, H, -1).transpose(1, 2))
    kf = apply_rope((xs @ wk).view(Bq, Tq, H, -1).transpose(1, 2))
    vf = (xs @ wv).view(Bq, Tq, H, -1).transpose(1, 2)
    full = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True).transpose(1, 2).reshape(Bq, Tq, D)
    close(decode_with_kv_cache(xs, wq, wk, wv, H), full)
    print("P12 KV-cache decode == full causal forward ok")

    lg4 = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=dt))
    keep = lambda o: torch.isfinite(o)[0].tolist()
    assert keep(filter_logits(lg4, k=2)) == [True, True, False, False]
    assert keep(filter_logits(lg4, p=0.75)) == [True, True, False, False]
    assert keep(filter_logits(lg4, p=0.85)) == [True, True, True, False]
    assert keep(filter_logits(lg4, p=0.1)) == [True, False, False, False]
    assert keep(filter_logits(lg4, temperature=0)) == [True, False, False, False]
    close(filter_logits(lg4, temperature=2.0).softmax(-1), (lg4 / 2).softmax(-1))
    print("P13 temperature/top-k/top-p ok (boundaries, T=0)")

    p_ref = torch.randn(5, dtype=dt, requires_grad=True); p_me = p_ref.detach().clone()
    opt = torch.optim.AdamW([p_ref], lr=0.1, weight_decay=0.1)
    mm, vv = torch.zeros(5, dtype=dt), torch.zeros(5, dtype=dt)
    for t in range(1, 6):
        g = torch.randn(5, dtype=dt)
        p_ref.grad = g.clone(); opt.step()
        adamw_step(p_me, g, mm, vv, t, lr=0.1, wd=0.1)
    close(p_me, p_ref.detach())
    print("P14 AdamW ok (5 steps vs torch.optim.AdamW)")

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
    print("P15 clip_grad_norm (vs torch) + warmup-cosine ok")

    a, b = torch.randn(5, 8, dtype=dt), torch.randn(5, 8, dtype=dt)
    close(kl_from_logits(a, b), F.kl_div(torch.log_softmax(b, -1), torch.log_softmax(a, -1), log_target=True, reduction="batchmean"))
    pl1, ql1 = torch.log_softmax(a[0], -1), torch.log_softmax(b[0], -1)
    xs_ = torch.multinomial(pl1.exp(), 400000, replacement=True, generator=torch.Generator().manual_seed(1))
    k1, k3 = kl_estimators(pl1[xs_], ql1[xs_])
    true = (pl1.exp() * (pl1 - ql1)).sum().item()
    assert abs(k1.mean().item() - true) < 0.02 and abs(k3.mean().item() - true) < 0.02 and (k3 >= 0).all()
    print(f"P16 KL ok (exact vs F.kl_div; MC k1 {k1.mean():.3f}, k3 {k3.mean():.3f}, true {true:.3f})")

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
    print("P17 sequence logprob + DPO + Bradley-Terry ok")

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
    print("P18 GRPO advantages + PPO clip ok")

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
    print(f"P19 MoE top-2 ok (vs per-token loop), aux {aux.item():.3f}, uniform aux = 1")

    W = torch.randn(5, 8, dtype=dt); A_ = torch.randn(2, 8, dtype=dt); B0 = torch.zeros(5, 2, dtype=dt)
    xl = torch.randn(3, 8, dtype=dt)
    close(lora_linear(xl, W, A_, B0, 16, 2), xl @ W.T)                            # B = 0 -> identical to base
    B1 = torch.randn(5, 2, dtype=dt)
    close(lora_linear(xl, W, A_, B1, 16, 2), xl @ lora_merge(W, A_, B1, 16, 2).T)
    print("P20 LoRA ok (zero-init identity, merge == unmerged)")

    text = "low lower lowest newer newest wider widest " * 5
    merges = bpe_train(text, 20)
    enc = bpe_encode(text, merges)
    assert bpe_decode(enc, merges) == text and len(enc) < len(text.encode())
    assert bpe_decode(bpe_encode("lowest 你好", merges), merges) == "lowest 你好"     # unseen / multibyte round trip
    assert merges[0][0] in {(ord("w"), ord("e")), (ord("e"), ord("s")), (ord("l"), ord("o")), (ord("o"), ord("w"))}
    print(f"P21 BPE ok ({len(text.encode())} bytes -> {len(enc)} tokens, round trip)")
    print("OK")


if __name__ == "__main__":
    test_all()
