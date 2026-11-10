"""Perplexity from scratch (torch + plain python), with tests.

Covers the usual interview variants:
  1. PPL from logits: shift by one, stable log-softmax by hand, ignore padding  (== F.cross_entropy reference)
  2. Batch aggregation: corpus PPL = exp(total NLL / total tokens), NOT mean of per-sequence PPLs
  3. Sliding-window (strided) PPL for texts longer than the model context (HF-style, context tokens masked out)
  4. n-gram LM perplexity with add-k smoothing (classic NLP version, pure python)
  5. Bits-per-token / bits-per-byte (tokenizer-independent comparison)

Interview checklist:
- PPL = exp( -1/N * sum_i log p(x_i | x_<i) ) = exp(cross-entropy in nats). base-2 version: 2^(bits per token)
- shift: logits[:, :-1] predict labels[:, 1:]; the first token has no prediction (unless there is a BOS)
- mask padding (and prompt tokens if you only want PPL on the response): sum NLL over real tokens, divide by their count
- aggregate over a corpus with token-weighted mean; mean of per-sequence PPL is biased toward short sequences and
  is not exp(mean NLL) (Jensen: mean(exp(x)) >= exp(mean(x)))
- sanity values: perfect model -> 1; uniform over V -> V; random-init LM -> ~V; PPL >= 1
- PPL depends on the tokenizer: different vocab -> not comparable. Use bits-per-byte: total_NLL / (ln2 * n_bytes)
- long text: sliding window with stride < window; each token is scored once, with as much left context as possible.
  stride == window (disjoint chunks) is cheaper but gives worse (higher) PPL because early tokens of each chunk
  see little context
- numerics: log_softmax = x - max - log(sum(exp(x - max))); accumulate NLL in float64 for big corpora
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 1. from logits
def log_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    return x - m - (x - m).exp().sum(dim=dim, keepdim=True).log()


def nll_sum_and_count(logits, input_ids, attention_mask=None):
    """logits (B,T,V) from a causal LM fed input_ids (B,T). Returns (sum NLL over predicted tokens, #tokens)."""
    logits = logits[:, :-1]                      # position t predicts token t+1
    targets = input_ids[:, 1:]
    logp = log_softmax(logits.double(), -1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # (B,T-1)
    if attention_mask is None:
        mask = torch.ones_like(targets, dtype=torch.double)
    else:
        mask = attention_mask[:, 1:].double()   # a target counts only if it is a real token
    return -(tok_logp * mask).sum(), mask.sum()


def perplexity(logits, input_ids, attention_mask=None):
    nll, n = nll_sum_and_count(logits, input_ids, attention_mask)
    return math.exp(nll.item() / n.item())


def per_sequence_ppl(logits, input_ids, attention_mask):
    return [perplexity(logits[i:i + 1], input_ids[i:i + 1], attention_mask[i:i + 1]) for i in range(len(input_ids))]


# ---------------------------------------------------------------- toy "model" with a limited context
class BigramLM(torch.nn.Module):
    """logits at position t depend only on token t -> a model with effective context 1. Has a hard max length
    so we can exercise sliding-window evaluation."""

    def __init__(self, V, max_len, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(V, V, generator=g, dtype=torch.double) * 2
        self.max_len = max_len

    def forward(self, ids):
        assert ids.shape[1] <= self.max_len, "input longer than context"
        return self.table[ids]                   # (B,T,V)


# ---------------------------------------------------------------- 3. sliding window
def sliding_window_ppl(model, ids, window, stride):
    """ids: (1, N). Each token from position 1 onward is scored exactly once; the first `window - trg_len` tokens of
    each window are context only (label masked). Mirrors the HF 'perplexity of fixed-length models' recipe."""
    N = ids.shape[1]
    total_nll, total_n = 0.0, 0
    prev_end = 1                                 # token 0 can never be predicted
    for begin in range(0, N, stride):
        end = min(begin + window, N)
        chunk = ids[:, begin:end]
        logits = model(chunk)[:, :-1].double()
        targets = chunk[:, 1:]
        tok_logp = log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        # target at chunk index j+1 corresponds to absolute position begin+j+1; score only new ones (>= prev_end)
        abs_pos = torch.arange(begin + 1, end)
        new = abs_pos >= prev_end
        total_nll += -tok_logp[new].sum().item()
        total_n += int(new.sum())
        prev_end = end
        if end == N:
            break
    # stride < window: every token 1..N-1 scored once. stride == window (disjoint chunks): the first token of each
    # chunk has no context inside the chunk, so it is never scored -> fewer tokens and a pessimistic PPL.
    return math.exp(total_nll / total_n), total_n


# ---------------------------------------------------------------- 4. n-gram LM
def train_ngram(tokens, n):
    ctx, full = Counter(), Counter()
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    for i in range(n - 1, len(padded)):
        h = tuple(padded[i - n + 1:i])
        ctx[h] += 1
        full[h + (padded[i],)] += 1
    vocab = set(tokens) | {"</s>", "<unk>"}
    return ctx, full, vocab


def ngram_ppl(model, tokens, n, k=1.0):
    ctx, full, vocab = model
    V = len(vocab)
    toks = [t if t in vocab else "<unk>" for t in tokens]
    padded = ["<s>"] * (n - 1) + toks + ["</s>"]
    nll = 0.0
    for i in range(n - 1, len(padded)):
        h = tuple(padded[i - n + 1:i])
        p = (full[h + (padded[i],)] + k) / (ctx[h] + k * V)      # add-k smoothing
        nll -= math.log(p)
    return math.exp(nll / (len(padded) - n + 1))


# ---------------------------------------------------------------- tests
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 3, 12, 50

    # 1: matches F.cross_entropy with ignore_index, padding handled
    logits = torch.randn(B, T, V, dtype=torch.double)
    ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 8:] = 0
    mask[2, 5:] = 0                              # right padding
    labels = ids.clone()
    labels[mask == 0] = -100
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    ours = perplexity(logits, ids, mask)
    assert abs(ours - ref.exp().item()) < 1e-9
    print(f"1. PPL from logits = {ours:.3f} == exp(F.cross_entropy) {ref.exp().item():.3f}")

    # sanity values
    uni = torch.zeros(1, T, V, dtype=torch.double)
    assert abs(perplexity(uni, ids[:1]) - V) < 1e-9
    perfect = torch.full((1, T, V), -1e4, dtype=torch.double)
    perfect[0, torch.arange(T - 1), ids[0, 1:]] = 1e4
    assert abs(perplexity(perfect, ids[:1]) - 1) < 1e-9
    print(f"   sanity: uniform -> {V}, perfect -> 1")

    # 2: corpus PPL vs mean of per-sequence PPL
    seqs = per_sequence_ppl(logits, ids, mask)
    mean_seq = sum(seqs) / len(seqs)
    print(f"2. corpus PPL {ours:.3f} vs mean per-seq PPL {mean_seq:.3f} (different! use token-weighted)")
    assert abs(ours - mean_seq) > 1e-6

    # 3: sliding window
    N = 200
    lm = BigramLM(V, max_len=32)
    long_ids = torch.randint(0, V, (1, N))
    full_ppl = math.exp(nll_sum_and_count(lm.table[long_ids], long_ids)[0].item() / (N - 1))
    for window, stride in [(32, 16), (32, 1), (32, 31)]:
        p, n = sliding_window_ppl(lm, long_ids, window, stride)
        # bigram model only needs 1 token of context, so any window >= 2 recovers the exact full-sequence PPL
        assert n == N - 1 and abs(p - full_ppl) < 1e-9, (window, stride, p, full_ppl)
    print(f"3. sliding window (stride 16/1/31) == full PPL {full_ppl:.3f}, every token scored once")
    p, n = sliding_window_ppl(lm, long_ids, 32, 32)
    assert n == N - math.ceil(N / 32)
    print(f"   disjoint chunks (stride == window): only {n}/{N - 1} tokens scored, PPL {p:.3f}")

    # 4: n-gram
    train = "the cat sat on the mat the dog sat on the log".split()
    test = "the cat sat on the log".split()
    for n in (1, 2, 3):
        m = train_ngram(train, n)
        print(f"4. {n}-gram add-1 PPL on test: {ngram_ppl(m, test, n):.3f}")
    m2 = train_ngram(train, 2)
    assert ngram_ppl(m2, test, 2) < ngram_ppl(train_ngram(train, 1), test, 1)
    # add-k distribution sums to 1 for any history
    ctx, full, vocab = m2
    for h in [("the",), ("zzz",)]:
        s = sum((full[h + (w,)] + 1) / (ctx[h] + len(vocab)) for w in vocab)
        assert abs(s - 1) < 1e-12

    # 5: bits per token / bits per byte
    text = "hello world, perplexity is exp of cross entropy"
    nll, n = nll_sum_and_count(logits, ids, mask)
    bpt = nll.item() / n.item() / math.log(2)
    bpb = nll.item() / math.log(2) / len(text.encode())   # pretend these tokens encode `text`
    assert abs(2 ** bpt - ours) < 1e-9
    print(f"5. bits/token {bpt:.3f} (2^bpt = PPL), bits/byte {bpb:.3f}")
    print("OK")
