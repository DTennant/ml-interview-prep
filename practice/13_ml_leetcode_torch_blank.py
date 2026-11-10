"""[PRACTICE / BLANK] Fill in every NotImplementedError, then run this file. Same tests as code/13.

LLM problems in LeetCode format (PyTorch): fixed signature -> implement -> hidden tests vs torch reference.

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
    raise NotImplementedError


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log of softmax along `dim`, computed directly (NOT log(softmax(x)), which underflows to -inf).

    log_softmax(x)_i = x_i - max(x) - log(sum_j exp(x_j - max(x))).
    Same shape as x. Must match torch.log_softmax(x, dim), including for very large / very negative logits.
    """
    raise NotImplementedError


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
    raise NotImplementedError


# P03 ------------------------------------------------------------------
def causal_lm_loss(logits, input_ids, attention_mask, prompt_lens=None):
    """Next-token prediction loss of a causal LM, plus perplexity.

    Args:
      logits: (B, T, V) model outputs for input_ids; logits[:, t] is the prediction for token t+1.
      input_ids: (B, T) long.
      attention_mask: (B, T) long/bool, 1 = real token, 0 = right padding.
      prompt_lens: optional (B,) long. If given, tokens at positions < prompt_lens[b] are prompt and are NOT
                   trained on (SFT: loss only on the response).
    Returns: (loss, ppl) where loss = mean NLL over all counted target tokens in the whole batch (token-weighted,
             not mean of per-sequence means) and ppl = exp(loss).
    A target token at position t (t >= 1) is counted iff attention_mask[b, t] == 1 and t >= prompt_lens[b].
    Must equal F.cross_entropy(logits[:, :-1].flatten, labels[:, 1:].flatten, ignore_index=-100) with the masked
    labels set to -100.
    """
    raise NotImplementedError


# P04 ------------------------------------------------------------------
def cross_entropy_backward(logits, target):
    """Gradient of mean cross-entropy w.r.t. the logits, derived by hand (no autograd).

    Args: logits (N, C) float, target (N,) long (no ignore_index here).
    Loss: L = (1/N) * sum_i -log softmax(logits_i)[target_i].
    Returns: dL/dlogits, shape (N, C) = (softmax(logits) - onehot(target)) / N.
    Must match logits.grad after F.cross_entropy(logits, target).backward().
    """
    raise NotImplementedError


# P05 ------------------------------------------------------------------
def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dimension.

    x: (..., D); weight, bias: (D,).
    y = (x - mean) / sqrt(var + eps) * weight + bias, mean/var over the last dim, var BIASED (divide by D).
    Must match F.layer_norm(x, (D,), weight, bias, eps).
    """
    raise NotImplementedError


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm (Llama) over the last dimension: no mean subtraction, no bias.

    x: (..., D); weight: (D,).  y = x / sqrt(mean(x^2, last dim) + eps) * weight.
    Must match F.rms_norm(x, (D,), weight, eps).
    """
    raise NotImplementedError


# P06 ------------------------------------------------------------------
def gelu(x, approximate="none"):
    """GELU activation, elementwise.

    approximate="none": exact,  0.5 * x * (1 + erf(x / sqrt(2))).
    approximate="tanh": GPT-2,  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Must match F.gelu(x, approximate=...).
    """
    raise NotImplementedError


def silu(x):
    """SiLU / Swish, elementwise: x * sigmoid(x). Must match F.silu."""
    raise NotImplementedError


def swiglu_ffn(x, w_gate, w_up, w_down):
    """Llama-style SwiGLU feed-forward block (no biases).

    x: (..., D); w_gate, w_up: (D, H); w_down: (H, D)   (weights are stored as (in, out), so use x @ w).
    y = (silu(x @ w_gate) * (x @ w_up)) @ w_down, shape (..., D).
    """
    raise NotImplementedError


# P07 ------------------------------------------------------------------
def dropout(x, p, training):
    """Inverted dropout.

    training=True: zero each element independently with probability p and scale survivors by 1 / (1 - p),
                   so E[output] == x. p == 1 -> all zeros (avoid division by zero). p == 0 -> x.
    training=False: return x unchanged (no scaling at eval time).
    Tested statistically: drop rate ~= p and mean preserved on a large tensor of ones.
    """
    raise NotImplementedError


# P08 ------------------------------------------------------------------
def sdpa(q, k, v, attn_mask=None, is_causal=False):
    """Scaled dot-product attention: softmax(q k^T / sqrt(d) + mask) v.

    Args:
      q: (..., T, d) queries; k, v: (..., S, d) keys / values (leading dims, e.g. (B, H), are batch dims).
      attn_mask: optional bool tensor broadcastable to (..., T, S); True = may attend, False = masked out.
      is_causal: if True, query i may only attend to keys j <= i + (S - T). When T < S (decoding with a cache)
                 the queries are the LAST T positions, so the mask is tril(ones(T, S), diagonal=S - T).
                 If both is_causal and attn_mask are given, a key must pass both.
    Returns: (..., T, d).
    Edge case: a query row with every key masked must output all zeros (plain softmax over all -inf gives NaN).
    Must match F.scaled_dot_product_attention for the non-degenerate cases.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def sinusoidal_pe(T, d, base=10000.0):
    """Original Transformer absolute positional encoding, returns (T, d) float64.

    pe[pos, 2i] = sin(pos / base^(2i/d)), pe[pos, 2i+1] = cos(pos / base^(2i/d)). Row 0 = [0, 1, 0, 1, ...].
    """
    raise NotImplementedError


# P10 ------------------------------------------------------------------
def multi_head_attention(x, w_qkv, b_qkv, w_o, b_o, n_heads, key_padding_mask=None, causal=False):
    """Standard multi-head self-attention, matching nn.MultiheadAttention(batch_first=True) weight layout.

    Args:
      x: (B, T, D) input. D must be divisible by n_heads; head dim Dh = D // n_heads.
      w_qkv: (3D, D) packed input projection = [W_q; W_k; W_v] stacked along dim 0 (PyTorch layout, out x in),
             so q = x @ W_q^T + b_q, etc.  b_qkv: (3D,) packed biases.
      w_o: (D, D) output projection, b_o: (D,)  ->  out = concat_heads @ w_o^T + b_o.
      n_heads: number of heads H.
      key_padding_mask: optional (B, T) bool, True = this KEY is padding and must not be attended to.
      causal: if True, position i attends only to positions <= i.
    Steps: project -> split into heads (B, H, T, Dh) -> scaled dot-product attention per head (scale 1/sqrt(Dh))
           -> merge heads back to (B, T, D) (transpose, then reshape) -> output projection.
    Returns: (B, T, D). No dropout. You may reuse your own sdpa from P08, not F.scaled_dot_product_attention.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# P15 ------------------------------------------------------------------
def clip_grad_norm(grads, max_norm, eps=1e-6):
    """Global-norm gradient clipping, like torch.nn.utils.clip_grad_norm_.

    grads: list of tensors of different shapes, modified IN PLACE.
    total = sqrt(sum over ALL tensors of sum(g^2))   (one norm across all params, not per tensor).
    If total > max_norm, scale every gradient by max_norm / (total + eps); otherwise leave unchanged.
    Returns: total norm BEFORE clipping (0-dim tensor).
    """
    raise NotImplementedError


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """Learning rate at 0-based `step` for linear warmup + cosine decay.

    step < warmup:  max_lr * (step + 1) / warmup        (so step warmup-1 reaches max_lr)
    otherwise:      progress = (step - warmup) / (total - warmup), clipped to [0, 1];
                    min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    So lr(warmup) = max_lr, lr at the midpoint of decay = (max_lr + min_lr) / 2, lr(>= total) = min_lr.
    """
    raise NotImplementedError


# P16 ------------------------------------------------------------------
def kl_from_logits(p_logits, q_logits):
    """Exact KL(P || Q) between two categorical distributions given as logits.

    p_logits, q_logits: (N, C). KL_n = sum_c P[n,c] * (log P[n,c] - log Q[n,c]); return the mean over N (scalar).
    Compute with log_softmax for stability. Note F.kl_div(input=log Q, target=log P) is the torch equivalent.
    """
    raise NotImplementedError


def kl_estimators(logp, logq):
    """Monte-Carlo estimators of KL(P || Q) from samples x ~ P (as used for the KL penalty in RLHF / GRPO).

    logp, logq: tensors of log P(x_i) and log Q(x_i) for samples x_i drawn from P (same shape).
    Return (k1, k3), per-sample estimates with the same shape:
      k1 = log p - log q                     (unbiased, high variance, can be negative)
      k3 = r - 1 - log r,  r = q / p          (unbiased, always >= 0, lower variance)
    The test checks that both means converge to the exact KL and that k3 >= 0.
    """
    raise NotImplementedError


# P17 ------------------------------------------------------------------
def sequence_logprob(logits, input_ids, loss_mask):
    """Total log-probability the model assigns to the selected tokens of each sequence (used by DPO / RL).

    logits: (B, T, V) for input_ids (B, T); loss_mask: (B, T), 1 on tokens that should be scored (the response).
    Token t (t >= 1) is predicted by logits[:, t-1]. Return (B,) = sum over t>=1 of loss_mask[b,t] * log p(ids[b,t]).
    SUM over tokens, not mean.
    """
    raise NotImplementedError


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimization loss.

    Inputs: (B,) sequence log-probs (from sequence_logprob) of the chosen / rejected responses under the policy
    (pi_*) and the frozen reference model (ref_*).
      z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
      loss = mean(-log sigmoid(z)) = mean(softplus(-z))
    Must be stable for |z| in the hundreds (do not compute log(sigmoid(z)) naively).
    Sanity: policy == reference -> loss = log 2.
    """
    raise NotImplementedError


def bradley_terry_loss(r_chosen, r_rejected):
    """Reward-model pairwise loss: mean(-log sigmoid(r_chosen - r_rejected)). r_*: (B,) scalar rewards.
    Must be finite even when r_rejected - r_chosen = 1600."""
    raise NotImplementedError


# P18 ------------------------------------------------------------------
def grpo_advantages(rewards, group_size, eps=1e-6):
    """GRPO advantage: normalise rewards within each group of samples from the same prompt.

    rewards: (num_prompts * group_size,) laid out group by group ([p0s0, p0s1, ..., p1s0, ...]).
    A = (r - mean_group) / (std_group + eps), std = unbiased (torch default). Returns the same shape.
    If all rewards in a group are equal, the advantages for that group are 0 (no learning signal).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# P20 ------------------------------------------------------------------
def lora_linear(x, W, A, B, alpha, r):
    """Linear layer with a LoRA adapter (no bias).

    x: (..., in); W: (out, in) frozen base weight (PyTorch layout); A: (r, in); B: (out, r).
    y = x @ W^T + (alpha / r) * (x @ A^T) @ B^T   — compute the low-rank path as two thin matmuls, do not form B @ A.
    With B initialised to zeros, the output equals the base layer exactly.
    """
    raise NotImplementedError


def lora_merge(W, A, B, alpha, r):
    """Fold the adapter into the base weight for inference: W' = W + (alpha / r) * B @ A, shape (out, in).
    x @ W'^T must equal lora_linear(x, W, A, B, alpha, r)."""
    raise NotImplementedError


# P21 ------------------------------------------------------------------
def bpe_train(text, n_merges):
    """Train a byte-level BPE tokenizer (GPT-2 / minbpe style, no regex pre-split, no special tokens).

    Start from the UTF-8 bytes of `text` (ids 0..255). Repeat n_merges times (stop early if < 2 tokens left):
    count all adjacent pairs, take the most frequent (ties -> the pair that appears first in the sequence),
    give it the next new id (256, 257, ...), and replace every non-overlapping occurrence left to right.
    Returns: list of merges in training order: [((id_a, id_b), new_id), ...].
    """
    raise NotImplementedError


def _merge(ids, pair, new):
    """Helper: replace each non-overlapping occurrence of `pair` in list `ids` (scanning left to right) by `new`."""
    raise NotImplementedError


def bpe_encode(text, merges):
    """Encode a string: UTF-8 bytes, then apply the merges in training order. Returns list of ints.
    Must work for text never seen in training (falls back to bytes), including multi-byte characters like 你好."""
    raise NotImplementedError


def bpe_decode(ids, merges):
    """Inverse of bpe_encode: build id -> bytes (0..255 are single bytes, each merge = concat of its pair),
    join the bytes, decode as UTF-8. decode(encode(s)) == s for any string s."""
    raise NotImplementedError


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

    # P10 MHA vs nn.MultiheadAttention
    D, H = 32, 8
    mha_ref = torch.nn.MultiheadAttention(D, H, batch_first=True, dtype=dt)
    xm_ = torch.randn(3, 7, D, dtype=dt)
    kpm = torch.zeros(3, 7, dtype=torch.bool); kpm[1, 5:] = True; kpm[2, 2:] = True
    args = (mha_ref.in_proj_weight.detach(), mha_ref.in_proj_bias.detach(),
            mha_ref.out_proj.weight.detach(), mha_ref.out_proj.bias.detach(), H)
    causal_m = torch.ones(7, 7, dtype=torch.bool).triu(1)          # nn.MHA: True = NOT allowed
    for kp, causal in [(None, False), (kpm, False), (None, True), (kpm, True)]:
        ref, _ = mha_ref(xm_, xm_, xm_, key_padding_mask=kp, attn_mask=causal_m if causal else None, need_weights=False)
        close(multi_head_attention(xm_, *args, key_padding_mask=kp, causal=causal), ref.detach())
    print("P10 MHA ok (vs nn.MultiheadAttention: plain / key padding / causal / both)")

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
