"""[PRACTICE / BLANK] Fill in every NotImplementedError, then run this file. Same tests as code/15.

Streaming / chunked / memory-efficient ML problems (PyTorch), in the same style as online entropy (14).

The common idea: you are not allowed to materialise the big thing (full softmax over a 150k vocab, the full
T x T attention matrix, the whole dataset, the whole batch). You read it in pieces, keep a small running
state, know how to RESCALE / MERGE that state, and the final answer must equal the naive version exactly.

Rules (same as 13): basic tensor ops only; do not call the torch function that implements the thing asked.
Every problem has a test at the bottom that compares against the naive / official implementation.

Problems
  S01 online KL(P || Q) between two logit vectors, over vocab chunks
  S02 fused linear + cross-entropy (never build the N x V logits), forward AND manual backward
  S03 FlashAttention forward: tiled attention with online softmax (+ return logsumexp)
  S04 streaming mean / variance (Welford) + merging two partial results (Chan)
  S05 chunked top-k and one-pass sampling over vocab chunks (Gumbel-max)
  S06 reservoir sampling: uniform k-of-n from a stream, and the weighted version
  S07 gradient accumulation with token-weighted loss (the classic micro-batch bug)
  S08 sequence packing: position_ids + block-diagonal causal mask
  S09 speculative decoding: the accept / reject verification step
  S10 causal linear attention: parallel form == recurrent (streaming state) form
"""
import math
import random

import torch
import torch.nn.functional as F


# S01 ------------------------------------------------------------------
def online_kl(p_logits, q_logits, chunk_size=1024):
    """KL(P || Q) where P = softmax(p_logits), Q = softmax(q_logits), along the last dim, reading the vocab in chunks.

    Input:  p_logits, q_logits: (..., V) unnormalised scores (same shape)
    Output: (...) KL in nats, >= 0. Must equal sum(P * (log P - log Q), -1).

    Derivation (a = p_logits, b = q_logits):
        log P_i = a_i - logZ_a,   log Q_i = b_i - logZ_b
        KL = sum_i P_i (a_i - b_i) - logZ_a + logZ_b
           = E_P[a - b] - logZ_a + logZ_b
    State per row:  m_a, s_a = sum exp(a - m_a), t = sum exp(a - m_a) * (a - b)   (P-side, like online entropy)
                    m_b, s_b = sum exp(b - m_b)                                     (Q-side, just a logsumexp)
    """
    raise NotImplementedError


# S02 ------------------------------------------------------------------
def fused_linear_cross_entropy(hidden, weight, targets, chunk_size=1024, ignore_index=-100):
    """Mean cross-entropy of (hidden @ weight.T) against targets, plus its gradients, WITHOUT building the
    (N, V) logits matrix. This is what Liger / "cut cross-entropy" do for the LM head.

    Input:  hidden  (N, D)  final hidden states (N = batch * seq tokens)
            weight  (V, D)  LM head / unembedding matrix
            targets (N,)    int64, ignore_index means "do not count this token"
    Output: loss (0-dim), grad_hidden (N, D), grad_weight (V, D)
            loss == F.cross_entropy(hidden @ weight.T, targets, ignore_index=...) and the grads equal autograd's.
    Peak extra memory is O(N * chunk_size) instead of O(N * V).

    Pass 1 (forward): for each vocab chunk compute logits_c = hidden @ W_c.T, update an online logsumexp, and
                      pick up the target logit when the target falls in this chunk.
    Pass 2 (backward): d loss / d logits = (softmax - onehot) / n_valid, on valid rows only.
                      Recompute logits_c chunk by chunk, softmax_c = exp(logits_c - lse), then
                      grad_hidden += dlogits_c @ W_c   and   grad_W_c = dlogits_c.T @ hidden.
    """
    raise NotImplementedError


# S03 ------------------------------------------------------------------
def flash_attention_forward(q, k, v, causal=False, block_q=16, block_k=16):
    """Tiled attention with online softmax: never builds the full (T, S) score matrix.

    Input:  q (B, H, T, Dh), k (B, H, S, Dh), v (B, H, S, Dv); causal requires T == S (query i sees keys 0..i)
    Output: out (B, H, T, Dv)  == softmax(q k^T / sqrt(Dh) [+ causal mask]) v
            lse (B, H, T)      == logsumexp of the scaled scores per query (what the backward pass needs)

    For each block of queries keep, per query row:
        m   = running max score,   l = running sum of exp(score - m),   acc = running sum of exp(score - m) * v
    For each block of keys: compute scores, new max, rescale l and acc by exp(m_old - m_new), add the block.
    At the end out = acc / l and lse = m + log l.
    """
    raise NotImplementedError


# S04 ------------------------------------------------------------------
class RunningMeanVar:
    """Streaming mean / variance per feature, numerically stable (Welford / Chan), mergeable.

    update(batch): batch is (n, D). merge(other): combine with another RunningMeanVar (e.g. from another GPU).
    Properties: mean (D,), var (D,) population variance (divide by N), var_unbiased (divide by N - 1).

    Keep count N, mean, and M2 = sum (x - mean)^2. Merging two groups A and B:
        delta = mean_B - mean_A,  N = N_A + N_B
        mean  = mean_A + delta * N_B / N
        M2    = M2_A + M2_B + delta^2 * N_A * N_B / N
    A batch is just a group whose mean/M2 we compute directly. Avoid the naive E[x^2] - E[x]^2: it cancels
    catastrophically when the mean is large compared to the std (e.g. 1e4 + small noise in float32).
    Used in: RunningMeanStd reward / observation normalisers in RL, BatchNorm running stats, dataset stats.
    """

    def __init__(self, dim, dtype=torch.float64):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=dtype)
        self.m2 = torch.zeros(dim, dtype=dtype)

    def _merge_stats(self, other_count, other_mean, other_m2):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def merge(self, other):
        raise NotImplementedError

    @property
    def var(self):
        raise NotImplementedError

    @property
    def var_unbiased(self):
        raise NotImplementedError


# S05 ------------------------------------------------------------------
def chunked_topk(logits, k, chunk_size=1024):
    """Top-k values and indices along the last dim, reading the vocab in chunks.

    Input:  logits (B, V); k <= V
    Output: values (B, k) sorted descending, indices (B, k) int64 global vocab indices. Values == torch.topk.
    Keep a running (B, k) best list; for each chunk, concat with the chunk and keep the top k again.
    (Same merge works for vocab-parallel top-k: each GPU sends its local top-k, then top-k of the union.)
    """
    raise NotImplementedError


def chunked_gumbel_sample(logits, chunk_size=1024, generator=None, temperature=1.0):
    """Sample one token per row from softmax(logits / temperature) in ONE pass over vocab chunks, never
    normalising. Gumbel-max trick: argmax_i (x_i / T + g_i), g_i = -log(-log(u_i)), u_i ~ Uniform(0, 1),
    is an exact sample from softmax(x / T). argmax is trivially mergeable across chunks / GPUs.

    Input:  logits (B, V); generator: optional torch.Generator for reproducibility
    Output: (B,) int64 sampled token ids
    """
    raise NotImplementedError


# S06 ------------------------------------------------------------------
def reservoir_sample(stream, k, rng=random):
    """Uniformly sample k items from a stream of unknown length in one pass, O(k) memory (Algorithm R).

    Input:  stream: any iterable; k >= 1; rng: object with .random() / .randint()
    Output: list of min(k, n) items; every item ends up in the sample with probability k / n.
    Item number i (0-based) is kept with probability k / (i + 1), replacing a uniformly random slot.
    """
    raise NotImplementedError


def weighted_reservoir_sample(stream, k, rng=random):
    """Weighted sampling without replacement from a stream of (item, weight) pairs (Efraimidis-Spirakis A-Res).

    Input:  stream of (item, weight > 0); k >= 1
    Output: list of min(k, n) items. For k = 1, item i is picked with probability w_i / sum(w).
    Give every item the key u^(1/w) with u ~ Uniform(0,1) and keep the k largest keys (a min-heap in practice).
    Equivalent in log space: key = log(u) / w, which is numerically nicer.
    """
    raise NotImplementedError


# S07 ------------------------------------------------------------------
def token_weighted_microbatch_losses(micro_logits, micro_labels, ignore_index=-100):
    """Losses to call .backward() on, one per micro-batch, so that the ACCUMULATED gradient equals the gradient
    of the full-batch mean token cross-entropy.

    Input:  micro_logits: list of (n_i, V) tensors; micro_labels: list of (n_i,) int64, ignore_index = padding
    Output: list of 0-dim losses; sum(losses) == F.cross_entropy(cat(logits), cat(labels), ignore_index=...)

    The bug everyone writes: loss_i = mean CE of micro-batch i, then loss_i / num_micro_batches. That weights
    every micro-batch equally, so a micro-batch with 3 real tokens counts as much as one with 300.
    Fix: count valid tokens over ALL micro-batches first, then loss_i = (sum of CE over micro-batch i) / total.
    (With data parallel the total must also be summed over ranks, e.g. all_reduce of the token count.)
    """
    raise NotImplementedError


# S08 ------------------------------------------------------------------
def packed_position_ids(lengths):
    """Position ids for several sequences packed into one row: restart from 0 at every sequence.

    Input:  lengths: list of ints, e.g. [3, 2, 4]
    Output: (sum(lengths),) int64, e.g. [0, 1, 2, 0, 1, 0, 1, 2, 3]
    """
    raise NotImplementedError


def packed_causal_mask(lengths):
    """Attention mask for packed sequences: causal inside each sequence, nothing across sequences.

    Input:  lengths: list of ints
    Output: (L, L) bool with L = sum(lengths); True = query row may attend to key column.
    Equivalent: same sequence id AND key position <= query position.
    (FlashAttention varlen takes cu_seqlens = [0, 3, 5, 9] instead of building this matrix.)
    """
    raise NotImplementedError


# S09 ------------------------------------------------------------------
def speculative_verify(draft_tokens, draft_probs, target_probs, generator=None):
    """One verification step of speculative decoding (Leviathan et al. / Chen et al.).

    Input:  draft_tokens (K,) int64: tokens proposed by the small model
            draft_probs  (K, V): q_i, the draft model's distribution at each proposed position
            target_probs (K+1, V): p_i, the big model's distribution at each position (one extra at the end)
    Output: list of accepted token ids, length between 1 and K+1.
            The output tokens are distributed EXACTLY as if sampled from the target model.

    For i = 0..K-1 with x = draft_tokens[i]:
        accept with probability min(1, p_i[x] / q_i[x])
        if rejected: sample from the residual norm(max(p_i - q_i, 0)), append it, and stop.
    If all K are accepted, sample one bonus token from p_K.
    """
    raise NotImplementedError


# S10 ------------------------------------------------------------------
def feature_map(x):
    """phi(x) = elu(x) + 1 > 0 (Katharopoulos et al. 2020)."""
    return F.elu(x) + 1.0


def linear_attention_parallel(q, k, v):
    """Causal linear attention, parallel (training) form, O(T^2) like normal attention but no softmax.

    Input:  q, k (B, T, D), v (B, T, Dv)
    Output: (B, T, Dv), out_t = sum_{s<=t} (phi(q_t) . phi(k_s)) v_s / sum_{s<=t} phi(q_t) . phi(k_s)
    """
    raise NotImplementedError


def linear_attention_recurrent(q, k, v):
    """Same output as linear_attention_parallel, computed token by token with an O(D * Dv) state (RNN form).

    State after step t:  S_t = sum_{s<=t} phi(k_s)^T v_s   (B, D, Dv)
                         z_t = sum_{s<=t} phi(k_s)          (B, D)
    out_t = phi(q_t) S_t / (phi(q_t) . z_t). Memory per step does not grow with T (unlike a KV cache).
    """
    raise NotImplementedError


# ======================================================================= tests
def close(a, b, tol=1e-6):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    assert torch.allclose(a, b.to(a.dtype), atol=tol, rtol=tol), (a - b).abs().max()


def test_all():
    torch.manual_seed(0)
    dt = torch.double

    # S01
    a = torch.randn(3, 5, 20_000) * 3
    b = torch.randn(3, 5, 20_000) * 3
    reference = (F.softmax(a, -1) * (F.log_softmax(a, -1) - F.log_softmax(b, -1))).sum(-1)
    for chunk in (777, 4096, 50_000):
        close(online_kl(a, b, chunk), reference, tol=1e-4)
    close(online_kl(a, a, 999), torch.zeros(3, 5), tol=1e-4)
    close(online_kl(a + 1000, b - 500, 999), reference, tol=1e-3)     # shift invariance, no overflow
    print("S01 online KL ok (several chunk sizes, KL(P||P) = 0, logits +-1000)")

    # S02
    N, D, V = 40, 16, 1000
    hidden = torch.randn(N, D, dtype=dt, requires_grad=True)
    weight = torch.randn(V, D, dtype=dt, requires_grad=True)
    targets = torch.randint(0, V, (N,)); targets[[3, 7, 11]] = -100
    ref_loss = F.cross_entropy(hidden @ weight.T, targets, ignore_index=-100)
    ref_loss.backward()
    loss, g_hidden, g_weight = fused_linear_cross_entropy(hidden.detach(), weight.detach(), targets, chunk_size=128)
    close(loss, ref_loss); close(g_hidden, hidden.grad); close(g_weight, weight.grad)
    print("S02 fused linear CE ok (loss + grad_hidden + grad_weight == autograd, with ignore_index)")

    # S03
    for causal, T, S, bq, bk in [(False, 37, 53, 8, 16), (True, 45, 45, 16, 8), (True, 45, 45, 7, 7)]:
        q = torch.randn(2, 3, T, 16, dtype=dt)
        k = torch.randn(2, 3, S, 16, dtype=dt)
        v = torch.randn(2, 3, S, 12, dtype=dt)
        out, lse = flash_attention_forward(q, k, v, causal=causal, block_q=bq, block_k=bk)
        close(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        scores = q @ k.transpose(-1, -2) / math.sqrt(16)
        if causal:
            scores = scores.masked_fill(~torch.ones(T, S, dtype=torch.bool).tril(), float("-inf"))
        close(lse, torch.logsumexp(scores, -1))
    print("S03 flash attention ok (== SDPA, causal + non-causal, uneven blocks, lse)")

    # S04
    data = torch.randn(1000, 6, dtype=dt) * 3 + 10_000.0
    left, right = RunningMeanVar(6), RunningMeanVar(6)
    for piece in torch.split(data[:600], 37):
        left.update(piece)
    for piece in torch.split(data[600:], 101):
        right.update(piece)
    left.merge(right)
    close(left.mean, data.mean(0)); close(left.var, data.var(0, unbiased=False))
    close(left.var_unbiased, data.var(0, unbiased=True))
    print("S04 Welford/Chan ok (uneven batches + merge == full-data mean/var)")

    # S05
    logits = torch.randn(4, 5000)
    values, indices = chunked_topk(logits, 50, chunk_size=333)
    ref_values, ref_indices = logits.topk(50, dim=1)
    close(values, ref_values); assert torch.equal(indices, ref_indices)
    small = torch.tensor([[2.0, 1.0, 0.0, -1.0, 0.5]]).repeat(200_000, 1)
    gen = torch.Generator().manual_seed(1)
    samples = chunked_gumbel_sample(small, chunk_size=2, generator=gen)
    frequency = torch.bincount(samples, minlength=5).float() / samples.numel()
    close(frequency, torch.softmax(small[0], 0), tol=5e-3)
    print("S05 chunked top-k == torch.topk; Gumbel-max sample frequencies match softmax")

    # S06
    rng = random.Random(0)
    counts = [0] * 10
    trials = 30_000
    for _ in range(trials):
        for item in reservoir_sample(range(10), 3, rng):
            counts[item] += 1
    assert all(abs(c / trials - 0.3) < 0.015 for c in counts), counts
    weights = [1.0, 2.0, 3.0, 4.0]
    picks = [0] * 4
    for _ in range(trials):
        picks[weighted_reservoir_sample(zip(range(4), weights), 1, rng)[0]] += 1
    assert all(abs(picks[i] / trials - weights[i] / 10) < 0.015 for i in range(4)), picks
    print("S06 reservoir ok (each item kept w.p. k/n; weighted k=1 picks w.p. w/sum(w))")

    # S07
    model = torch.nn.Linear(8, 11).double()
    xs = [torch.randn(n, 8, dtype=dt) for n in (5, 20, 3)]
    ys = [torch.randint(0, 11, (x.shape[0],)) for x in xs]
    ys[1][:12] = -100                                                 # micro-batches with very different #tokens
    full_loss = F.cross_entropy(model(torch.cat(xs)), torch.cat(ys), ignore_index=-100)
    full_grad = torch.autograd.grad(full_loss, model.weight)[0]
    model.zero_grad()
    for loss in token_weighted_microbatch_losses([model(x) for x in xs], ys):
        loss.backward()
    close(model.weight.grad, full_grad)
    model.zero_grad()
    for x, y in zip(xs, ys):                                          # the buggy version
        (F.cross_entropy(model(x), y, ignore_index=-100) / len(xs)).backward()
    assert not torch.allclose(model.weight.grad, full_grad)
    print("S07 token-weighted grad accumulation == full batch (naive mean / n_micro does not)")

    # S08
    lengths = [3, 2, 4]
    assert packed_position_ids(lengths).tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    mask = packed_causal_mask(lengths)
    L = sum(lengths)
    q, k, v = (torch.randn(1, 2, L, 8, dtype=dt) for _ in range(3))
    packed_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    pieces, start = [], 0
    for n in lengths:
        sl = slice(start, start + n)
        pieces.append(F.scaled_dot_product_attention(q[:, :, sl], k[:, :, sl], v[:, :, sl], is_causal=True))
        start += n
    close(packed_out, torch.cat(pieces, dim=2))
    print("S08 packing ok (packed attention == each sequence on its own)")

    # S09
    p = torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]], dtype=dt)  # target p_0, p_1
    q0 = torch.tensor([[0.1, 0.6, 0.1, 0.2]], dtype=dt)                           # draft q_0
    gen = torch.Generator().manual_seed(0)
    first_tokens = torch.zeros(4, dtype=dt)
    trials = 40_000
    for _ in range(trials):
        draft = torch.multinomial(q0[0], 1, generator=gen)
        first_tokens[speculative_verify(draft, q0, p, generator=gen)[0]] += 1
    close(first_tokens / trials, p[0], tol=0.012)
    print("S09 speculative decoding ok (first output token ~ target p, not draft q)")

    # S10
    q, k, v = torch.randn(2, 30, 8, dtype=dt), torch.randn(2, 30, 8, dtype=dt), torch.randn(2, 30, 5, dtype=dt)
    close(linear_attention_parallel(q, k, v), linear_attention_recurrent(q, k, v))
    print("S10 linear attention ok (parallel == recurrent)")
    print("OK")


if __name__ == "__main__":
    test_all()
