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
    """Numerically stable log-softmax written by hand.

        log_softmax(x)_i = x_i - log(sum_j exp(x_j))
                         = (x_i - m) - log(sum_j exp(x_j - m))     with m = max_j x_j

    Subtracting the max first means the largest exponent is exp(0) = 1, so exp never overflows.

    Input:  x (..., V) logits
    Output: (..., V) log-probabilities (each row's exp sums to 1)
    """
    # Step 1: shift so the largest logit in each row is 0
    row_max = x.max(dim=dim, keepdim=True).values                     # (..., 1)
    shifted = x - row_max                                             # (..., V)  all <= 0

    # Step 2: log of the normaliser, computed on the shifted (safe) values
    log_normaliser = shifted.exp().sum(dim=dim, keepdim=True).log()   # (..., 1)

    # Step 3: log-probabilities
    return shifted - log_normaliser                                   # (..., V)


def nll_sum_and_count(logits, input_ids, attention_mask=None):
    """Total negative log-likelihood of the predicted tokens, and how many tokens were predicted.

    A causal LM's output at position t is a prediction for token t+1, so we drop the last logit and the first
    token. Padding tokens (attention_mask == 0) are not counted.

    Input:  logits (B, T, V) from a causal LM fed input_ids (B, T); attention_mask (B, T) of 0/1 or None
    Output: (nll_sum, num_tokens) -- two 0-dim float64 tensors
    """
    # Step 1: align predictions with targets (position t predicts token t+1)
    pred_logits = logits[:, :-1]                                      # (B, T-1, V)
    targets = input_ids[:, 1:]                                        # (B, T-1)

    # Step 2: log-probabilities over the vocab; float64 so summing over many tokens stays accurate
    log_probs = log_softmax(pred_logits.double(), dim=-1)             # (B, T-1, V)

    # Step 3: pick out the log-prob assigned to the actual next token
    target_index = targets.unsqueeze(-1)                              # (B, T-1, 1)
    target_log_probs = log_probs.gather(-1, target_index)             # (B, T-1, 1)
    target_log_probs = target_log_probs.squeeze(-1)                   # (B, T-1)

    # Step 4: which targets count. A target is counted only if it is a real (non-padding) token.
    if attention_mask is None:
        mask = torch.ones_like(targets, dtype=torch.double)           # (B, T-1)
    else:
        mask = attention_mask[:, 1:].double()                         # (B, T-1)

    # Step 5: sum NLL over the counted tokens
    masked_log_probs = target_log_probs * mask                        # (B, T-1)  padding contributes 0
    nll_sum = -masked_log_probs.sum()                                 # ()
    num_tokens = mask.sum()                                           # ()
    return nll_sum, num_tokens


def perplexity(logits, input_ids, attention_mask=None):
    """Perplexity = exp(average NLL per predicted token), averaged over ALL tokens in the batch.

    Input:  logits (B, T, V); input_ids (B, T); attention_mask (B, T) or None
    Output: float
    """
    nll_sum, num_tokens = nll_sum_and_count(logits, input_ids, attention_mask)
    mean_nll = nll_sum.item() / num_tokens.item()     # cross-entropy in nats per token
    return math.exp(mean_nll)


def per_sequence_ppl(logits, input_ids, attention_mask):
    """Perplexity of each sequence on its own (to show it differs from corpus PPL).

    Input:  logits (B, T, V); input_ids (B, T); attention_mask (B, T)
    Output: list of B floats
    """
    results = []
    num_sequences = len(input_ids)
    for i in range(num_sequences):
        # slice with i:i+1 to keep the batch dimension -> shapes (1, T, V), (1, T), (1, T)
        seq_ppl = perplexity(logits[i:i + 1], input_ids[i:i + 1], attention_mask[i:i + 1])
        results.append(seq_ppl)
    return results


# ---------------------------------------------------------------- toy "model" with a limited context
class BigramLM(torch.nn.Module):
    """logits at position t depend only on token t -> a model with effective context 1. Has a hard max length
    so we can exercise sliding-window evaluation.

    Input:  ids (B, T) with T <= max_len
    Output: logits (B, T, V)
    """

    def __init__(self, V, max_len, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        # table[a] = logits for the token that follows token a
        self.table = torch.randn(V, V, generator=generator, dtype=torch.double) * 2   # (V, V)
        self.max_len = max_len

    def forward(self, ids):
        assert ids.shape[1] <= self.max_len, "input longer than context"
        return self.table[ids]                                        # (B, T, V)


# ---------------------------------------------------------------- 3. sliding window
def sliding_window_ppl(model, ids, window, stride):
    """ids: (1, N). Each token from position 1 onward is scored exactly once; the first `window - trg_len` tokens of
    each window are context only (label masked). Mirrors the HF 'perplexity of fixed-length models' recipe.

    The window slides forward by `stride`. Tokens already scored by an earlier window are re-used only as
    context; only the NEW tokens (absolute position >= prev_end) contribute to the loss.

    Input:  model mapping (1, L) ids -> (1, L, V) logits, L <= window; ids (1, N)
    Output: (perplexity float, number of scored tokens int)
    """
    N = ids.shape[1]
    total_nll = 0.0
    total_n = 0
    prev_end = 1        # first absolute position not yet scored; token 0 can never be predicted

    for begin in range(0, N, stride):
        end = min(begin + window, N)

        # Step 1: run the model on this window
        chunk = ids[:, begin:end]                                     # (1, L)  L = end - begin
        logits = model(chunk)                                         # (1, L, V)

        # Step 2: log-prob of each next token inside the window (position j predicts j+1)
        pred_logits = logits[:, :-1].double()                         # (1, L-1, V)
        targets = chunk[:, 1:]                                        # (1, L-1)
        log_probs = log_softmax(pred_logits, dim=-1)                  # (1, L-1, V)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1))   # (1, L-1, 1)
        target_log_probs = target_log_probs.squeeze(-1)[0]            # (L-1,)

        # Step 3: keep only targets we haven't scored yet. Target j of this window is the token at
        # absolute position begin + 1 + j.
        target_abs_pos = torch.arange(begin + 1, end)                 # (L-1,)
        is_new = target_abs_pos >= prev_end                           # (L-1,) bool

        # Step 4: accumulate
        total_nll += -target_log_probs[is_new].sum().item()
        total_n += int(is_new.sum())
        prev_end = end

        if end == N:
            break

    # stride < window: every token 1..N-1 scored once. stride == window (disjoint chunks): the first token of each
    # chunk has no context inside the chunk, so it is never scored -> fewer tokens and a pessimistic PPL.
    ppl = math.exp(total_nll / total_n)
    return ppl, total_n


# ---------------------------------------------------------------- 4. n-gram LM
def train_ngram(tokens, n):
    """Count n-grams and their (n-1)-token histories.

    The sequence is padded with n-1 "<s>" at the start (so the first word has a history) and "</s>" at the end
    (so the model learns where sentences stop).

    Input:  tokens (list of str), n (int)
    Output: (ctx_counts, ngram_counts, vocab)
            ctx_counts[history]           = how often `history` (tuple of n-1 tokens) was followed by anything
            ngram_counts[history + (w,)]  = how often history was followed by w
            vocab                         = training tokens + "</s>" + "<unk>"
    """
    ctx_counts = Counter()
    ngram_counts = Counter()
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]

    for i in range(n - 1, len(padded)):
        history = tuple(padded[i - n + 1:i])      # the n-1 tokens before position i (empty tuple for unigrams)
        word = padded[i]
        ctx_counts[history] += 1
        ngram_counts[history + (word,)] += 1

    vocab = set(tokens) | {"</s>", "<unk>"}
    return ctx_counts, ngram_counts, vocab


def ngram_ppl(model, tokens, n, k=1.0):
    """Perplexity of an n-gram model with add-k smoothing on a test token list.

        p(w | h) = (count(h, w) + k) / (count(h) + k * V)

    Adding k to every count gives unseen n-grams non-zero probability (otherwise one unseen n-gram -> PPL = inf).

    Input:  model = output of train_ngram; tokens (list of str); n; k smoothing constant
    Output: float perplexity
    """
    ctx_counts, ngram_counts, vocab = model
    V = len(vocab)

    # Step 1: map out-of-vocabulary words to <unk>, then pad exactly like in training
    mapped = []
    for t in tokens:
        if t in vocab:
            mapped.append(t)
        else:
            mapped.append("<unk>")
    padded = ["<s>"] * (n - 1) + mapped + ["</s>"]

    # Step 2: sum NLL over every predicted position (all real tokens plus the final </s>)
    nll = 0.0
    for i in range(n - 1, len(padded)):
        history = tuple(padded[i - n + 1:i])
        word = padded[i]
        numerator = ngram_counts[history + (word,)] + k
        denominator = ctx_counts[history] + k * V
        prob = numerator / denominator
        nll -= math.log(prob)

    # Step 3: average and exponentiate
    num_predictions = len(padded) - n + 1
    return math.exp(nll / num_predictions)


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
    full_nll, _ = nll_sum_and_count(lm.table[long_ids], long_ids)
    full_ppl = math.exp(full_nll.item() / (N - 1))
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
    bpt = nll.item() / n.item() / math.log(2)                # nats -> bits: divide by ln 2
    bpb = nll.item() / math.log(2) / len(text.encode())      # pretend these tokens encode `text`
    assert abs(2 ** bpt - ours) < 1e-9
    print(f"5. bits/token {bpt:.3f} (2^bpt = PPL), bits/byte {bpb:.3f}")
    print("OK")
