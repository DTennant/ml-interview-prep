"""Online (streaming / chunked) entropy, logsumexp and cross-entropy, with tests.

Problem: the vocab V is huge (e.g. 150k) and the logits come in chunks (or we do not want to
materialise the full softmax). Compute in ONE pass over the logits, with O(1) extra memory per row:
  1. logsumexp(x)                      -- the "online softmax" trick (same as in FlashAttention)
  2. entropy of softmax(x)             -- H = logsumexp(x) - E_p[x]
  3. cross-entropy for a target index  -- CE = logsumexp(x) - x[target]
  4. merging two partial states        -- lets you split the vocab across threads / GPUs (vocab parallel)
  5. bonus: entropy of an empirical distribution from a stream of samples (counts), O(1) per update

Derivation for the entropy:
    p_i = exp(x_i) / Z,   Z = sum_j exp(x_j),   log Z = logsumexp(x)
    H   = -sum_i p_i log p_i
        = -sum_i p_i (x_i - log Z)
        = log Z - sum_i p_i x_i
So we only need three running numbers per row:
    m = max logit seen so far                      (for numerical stability)
    s = sum_i exp(x_i - m)                         (normaliser, relative to m)
    t = sum_i exp(x_i - m) * x_i                   (unnormalised weighted sum of logits)
and at the end:
    log Z       = m + log(s)
    E_p[x]      = t / s
    H           = m + log(s) - t / s

When the running max changes from m_old to m_new, every old term exp(x_i - m_old) has to become
exp(x_i - m_new) = exp(x_i - m_old) * exp(m_old - m_new). So both s and t are multiplied by the
same factor exp(m_old - m_new). (x_i itself does not change, so t uses the same rescale as s.)

Interview checklist:
- why subtract the max: exp(x) overflows for x > ~88 in float32; after subtracting the max all exponents are <= 0
- start with m = -inf, s = t = 0; exp(-inf - m_new) = 0 so the first rescale is a no-op. Edge case: if every logit seen
  so far is -inf (fully masked), -inf - (-inf) = nan; replace -inf by a large negative number like -1e9 to avoid it
- complexity: time O(V), extra memory O(1) per row (never stores the V-dim probabilities)
- accumulate in float32 even if logits are bf16
- states are associative to merge -> tree reduction, or one partial state per GPU then all-reduce
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 1 + 2. online logsumexp and entropy
def online_entropy(logits, chunk_size=1024):
    """Entropy (in nats) of softmax(logits) along the last dim, reading the last dim in chunks.

    Input:  logits (..., V) unnormalised scores; chunk_size = how many vocab entries to read at a time
    Output: entropy (...), one number per row. Same as -(softmax(x) * log_softmax(x)).sum(-1)
    """
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    device = logits.device

    # Running statistics for every row
    running_max = torch.full(batch_shape, float("-inf"), device=device)   # m
    running_sum_exp = torch.zeros(batch_shape, device=device)             # s = sum exp(x - m)
    running_weighted_sum = torch.zeros(batch_shape, device=device)        # t = sum exp(x - m) * x

    for start in range(0, vocab_size, chunk_size):
        chunk = logits[..., start:start + chunk_size].float()             # (..., C)

        # Step 1: the new max after seeing this chunk
        chunk_max = chunk.max(dim=-1).values                              # (...)
        new_max = torch.maximum(running_max, chunk_max)                   # (...)

        # Step 2: rescale the old statistics from the old max to the new max
        #         at the start running_max = -inf, so scale = exp(-inf) = 0 (correct: nothing accumulated yet)
        scale = torch.exp(running_max - new_max)                          # (...)
        running_sum_exp = running_sum_exp * scale
        running_weighted_sum = running_weighted_sum * scale

        # Step 3: add this chunk's contribution, measured relative to the new max
        exp_chunk = torch.exp(chunk - new_max.unsqueeze(-1))              # (..., C)  all <= 1
        running_sum_exp = running_sum_exp + exp_chunk.sum(dim=-1)
        running_weighted_sum = running_weighted_sum + (exp_chunk * chunk).sum(dim=-1)

        running_max = new_max

    # Step 4: H = log Z - E_p[x]
    log_partition = running_max + torch.log(running_sum_exp)              # log Z = m + log s
    expected_logit = running_weighted_sum / running_sum_exp               # E_p[x] = t / s
    entropy = log_partition - expected_logit
    return entropy


def online_logsumexp(logits, chunk_size=1024):
    """logsumexp along the last dim, reading it in chunks. Only needs (m, s).

    Input:  logits (..., V)
    Output: (...)  == torch.logsumexp(logits, dim=-1)
    """
    batch_shape = logits.shape[:-1]
    running_max = torch.full(batch_shape, float("-inf"), device=logits.device)
    running_sum_exp = torch.zeros(batch_shape, device=logits.device)

    for start in range(0, logits.shape[-1], chunk_size):
        chunk = logits[..., start:start + chunk_size].float()
        new_max = torch.maximum(running_max, chunk.max(dim=-1).values)
        running_sum_exp = running_sum_exp * torch.exp(running_max - new_max)
        running_sum_exp = running_sum_exp + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        running_max = new_max

    return running_max + torch.log(running_sum_exp)


# ---------------------------------------------------------------- 3. online cross-entropy
def online_cross_entropy(logits, targets, chunk_size=1024):
    """Mean cross-entropy loss, reading the vocab in chunks.

        CE_row = logsumexp(x) - x[target]

    The target logit lives in exactly one chunk, so we pick it up when we pass that chunk.

    Input:  logits (N, V); targets (N,) int64 class indices
    Output: 0-dim tensor, == F.cross_entropy(logits, targets)
    """
    num_rows, vocab_size = logits.shape
    running_max = torch.full((num_rows,), float("-inf"), device=logits.device)
    running_sum_exp = torch.zeros(num_rows, device=logits.device)
    target_logit = torch.zeros(num_rows, device=logits.device)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = logits[:, start:end].float()                              # (N, C)

        # online logsumexp update (same as above)
        new_max = torch.maximum(running_max, chunk.max(dim=-1).values)
        running_sum_exp = running_sum_exp * torch.exp(running_max - new_max)
        running_sum_exp = running_sum_exp + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        running_max = new_max

        # grab the target logit if the target index falls inside this chunk
        in_this_chunk = (targets >= start) & (targets < end)              # (N,) bool
        local_index = (targets - start).clamp(0, end - start - 1)         # (N,) safe index into the chunk
        picked = chunk.gather(1, local_index.unsqueeze(1)).squeeze(1)     # (N,)
        target_logit = torch.where(in_this_chunk, picked, target_logit)

    log_partition = running_max + torch.log(running_sum_exp)
    loss_per_row = log_partition - target_logit
    return loss_per_row.mean()


# ---------------------------------------------------------------- 4. merging partial states
def partial_state(chunk):
    """(m, s, t) for one chunk of logits (..., C). Treat each chunk as computed on a different worker."""
    chunk = chunk.float()
    m = chunk.max(dim=-1).values
    e = torch.exp(chunk - m.unsqueeze(-1))
    s = e.sum(dim=-1)
    t = (e * chunk).sum(dim=-1)
    return m, s, t


def merge_states(state_a, state_b):
    """Combine two partial states into one. Associative + commutative -> any reduction order works.

    Both states are rescaled to the larger of the two maxes, then added.
    """
    m_a, s_a, t_a = state_a
    m_b, s_b, t_b = state_b
    m = torch.maximum(m_a, m_b)
    scale_a = torch.exp(m_a - m)
    scale_b = torch.exp(m_b - m)
    s = s_a * scale_a + s_b * scale_b
    t = t_a * scale_a + t_b * scale_b
    return m, s, t


def entropy_from_state(state):
    m, s, t = state
    return m + torch.log(s) - t / s


# ---------------------------------------------------------------- 5. empirical entropy from a stream of samples
class StreamingEntropy:
    """Entropy (nats) of the empirical distribution of a stream of discrete items, O(1) per update.

        H = -sum_c (n_c / N) log(n_c / N) = log N - (1/N) sum_c n_c log n_c

    Keep N and S = sum_c n_c log n_c. When item c arrives, only its own term changes:
        S += (n_c + 1) log(n_c + 1) - n_c log n_c
    """

    def __init__(self):
        self.counts = Counter()
        self.total = 0                 # N
        self.sum_n_log_n = 0.0         # S

    @staticmethod
    def _n_log_n(n):
        return n * math.log(n) if n > 0 else 0.0

    def add(self, item):
        old_count = self.counts[item]
        new_count = old_count + 1
        self.sum_n_log_n += self._n_log_n(new_count) - self._n_log_n(old_count)
        self.counts[item] = new_count
        self.total += 1

    def entropy(self):
        if self.total == 0:
            return 0.0
        return math.log(self.total) - self.sum_n_log_n / self.total


# ---------------------------------------------------------------- tests
if __name__ == "__main__":
    torch.manual_seed(0)

    def reference_entropy(x):
        return -(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)).sum(dim=-1)

    # 1 + 2: entropy and logsumexp match torch, for chunk sizes that do / do not divide V
    logits = torch.randn(4, 7, 50_000) * 3
    for chunk_size in (1, 999, 1024, 50_000, 100_000):
        if chunk_size == 1:
            small = logits[:, :, :300]
            assert torch.allclose(online_entropy(small, 1), reference_entropy(small), atol=1e-4)
            continue
        assert torch.allclose(online_entropy(logits, chunk_size), reference_entropy(logits), atol=1e-4)
        assert torch.allclose(online_logsumexp(logits, chunk_size), torch.logsumexp(logits, -1), atol=1e-4)
    print("1+2. online entropy / logsumexp == torch for several chunk sizes")

    # numerical stability: huge logits that would overflow a naive exp
    big = torch.randn(3, 10_000) * 5 + 1000.0
    assert torch.isfinite(online_entropy(big)).all()
    assert torch.allclose(online_entropy(big), reference_entropy(big), atol=1e-3)
    # the max ordering does not matter: a sorted-ascending row forces a rescale at every chunk
    ascending = torch.sort(torch.randn(2, 20_000) * 4, dim=-1).values
    assert torch.allclose(online_entropy(ascending, 512), reference_entropy(ascending), atol=1e-4)
    # sanity values: uniform -> log V, one-hot-ish -> ~0
    V = 4096
    assert abs(online_entropy(torch.zeros(1, V)).item() - math.log(V)) < 1e-4
    peaked = torch.full((1, V), -1e4)
    peaked[0, 17] = 0.0
    assert online_entropy(peaked).item() < 1e-4
    print("     stable at logits ~1000, uniform -> log V, peaked -> 0")

    # 3: cross-entropy
    x = torch.randn(64, 30_000) * 2
    y = torch.randint(0, 30_000, (64,))
    ours = online_cross_entropy(x, y, chunk_size=777)
    assert torch.allclose(ours, F.cross_entropy(x, y), atol=1e-4)
    print(f"3. online CE {ours.item():.4f} == F.cross_entropy")

    # 4: split the vocab into 8 shards ("8 GPUs"), merge in two different orders
    shards = torch.chunk(logits, 8, dim=-1)
    states = [partial_state(s) for s in shards]
    left_to_right = states[0]
    for st in states[1:]:
        left_to_right = merge_states(left_to_right, st)
    tree = states
    while len(tree) > 1:
        tree = [merge_states(tree[i], tree[i + 1]) for i in range(0, len(tree), 2)]
    assert torch.allclose(entropy_from_state(left_to_right), reference_entropy(logits), atol=1e-4)
    assert torch.allclose(entropy_from_state(tree[0]), reference_entropy(logits), atol=1e-4)
    print("4. merged shard states (sequential and tree reduction) == full entropy")

    # 5: streaming empirical entropy
    stream = torch.randint(0, 50, (20_000,)).tolist()
    tracker = StreamingEntropy()
    for item in stream:
        tracker.add(item)
    counts = torch.bincount(torch.tensor(stream)).double()
    p = counts[counts > 0] / counts.sum()
    assert abs(tracker.entropy() - (-(p * p.log()).sum().item())) < 1e-9
    print(f"5. streaming empirical entropy {tracker.entropy():.4f} (max log 50 = {math.log(50):.4f})")
    print("OK")
