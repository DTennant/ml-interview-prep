"""Debug drill: a buggy training loop with 6 planted bugs, followed by the fixed version.

Try first: read `buggy_train` and list every bug before scrolling to `fixed_train`.
Planted bugs:
  1. optimizer.zero_grad() missing  -> gradients accumulate across steps
  2. softmax applied before F.cross_entropy (which already does log_softmax) -> double softmax, weak grads
  3. labels flattened in a different order than logits (yb.t().reshape(-1)) -> silent misalignment
  4. causal mask inverted (masks the past instead of the future) -> model can cheat / or can't learn
  5. model.eval() / torch.no_grad() missing in evaluation (dropout active during eval)
  6. data never shuffled + learning rate 10x too high
Common real-world extras to mention: NaN from log(0), wrong dtype for labels (float vs long),
forgetting .detach() when logging (memory leak), loss averaged twice, train/eval data leakage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB_SIZE = 10
BATCH_SIZE = 64


def make_data(n=2000, T=8, V=10):
    """Build a toy "predict the previous token" dataset.

    Task: at position t the target is the input token at position t-1
    (position 0 has target 0). Solving it requires attending to the PAST only,
    which is exactly what a correct causal mask allows.

    Args:
        n: number of sequences.
        T: sequence length.
        V: vocabulary size. Input tokens are drawn from 1..V-1 (0 is reserved
           as the "no previous token" label for position 0).

    Returns:
        x: (n, T) long tensor of input tokens.
        y: (n, T) long tensor of targets, y[:, t] = x[:, t-1], y[:, 0] = 0.
    """
    # Step 1: random input tokens in [1, V).
    x = torch.randint(1, V, (n, T))                                  # (n, T)

    # Step 2: target = input shifted right by one position.
    first_column_label = torch.zeros(n, 1, dtype=torch.long)         # (n, 1) label for position 0
    previous_tokens = x[:, :-1]                                      # (n, T-1) tokens 0..T-2
    y = torch.cat([first_column_label, previous_tokens], dim=1)      # (n, T)
    return x, y


class TinyAttnModel(nn.Module):
    """One-layer, single-head self-attention model with a residual connection.

    forward: tokens (B, T) -> logits (B, T, V)

    If `invert_mask=True` the causal mask is flipped (BUG 4): each position can
    only see the FUTURE (strictly later tokens) instead of itself + the past,
    so the "copy previous token" task becomes impossible.
    """

    def __init__(self, V=10, d=32, T=8, invert_mask=False):
        """Create layers.

        Args:
            V: vocab size.  d: model width.  T: max sequence length.
            invert_mask: plant BUG 4 (inverted causal mask) when True.
        """
        super().__init__()
        # NOTE: layers are created in this exact order so the random init
        # (and therefore the printed accuracies) stays reproducible.
        self.emb = nn.Embedding(V, d)            # token embedding
        self.pos = nn.Embedding(T, d)            # learned position embedding
        self.qkv = nn.Linear(d, 3 * d)           # projects h to [q | k | v] in one matmul
        self.out = nn.Linear(d, V)               # final projection to vocab logits
        self.drop = nn.Dropout(0.3)
        self.invert = invert_mask

    def forward(self, x):
        """Run single-head causal self-attention.

        Args:
            x: (B, T) long tensor of token ids.
        Returns:
            logits: (B, T, V) unnormalised scores (NOT softmaxed).

        Formula: att = softmax(mask(Q K^T / sqrt(d))),  out = W_out(dropout(att V) + h)
        """
        B, T = x.shape

        # Step 1: token + position embeddings.
        positions = torch.arange(T)                                  # (T,)
        h = self.emb(x) + self.pos(positions)                        # (B, T, d)

        # Step 2: project to queries, keys, values.
        qkv = self.qkv(h)                                            # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)                               # each (B, T, d)

        # Step 3: scaled dot-product scores. Scale by sqrt(d) so the softmax
        # does not saturate when d is large.
        d_head = q.shape[-1]
        scores = q @ k.transpose(-1, -2)                             # (B, T, T)
        scores = scores / d_head ** 0.5                              # (B, T, T)

        # Step 4: causal mask. allowed[i, j] = True means "query i may look at key j".
        # Correct version: lower-triangular -> position i sees positions 0..i.
        allowed = torch.ones(T, T, dtype=torch.bool).tril()         # (T, T)
        if self.invert:
            # BUG 4: flipping the mask means position i only sees j > i (the future).
            # For "copy previous token" this hides exactly the token we need.
            allowed = ~allowed

        # Step 5: block disallowed positions with a huge negative number, then softmax.
        scores = scores.masked_fill(~allowed, -1e9)                  # (B, T, T)
        attn_weights = scores.softmax(dim=-1)                        # (B, T, T), rows sum to 1

        # Step 6: weighted sum of values, dropout, residual, project to vocab.
        attended = attn_weights @ v                                  # (B, T, d)
        attended = self.drop(attended)                               # (B, T, d)
        residual = attended + h                                      # (B, T, d)
        logits = self.out(residual)                                  # (B, T, V)
        return logits


def accuracy(model, x, y, fixed):
    """Token-level accuracy of `model` on (x, y).

    Args:
        model: TinyAttnModel.
        x, y: (N, T) inputs and targets.
        fixed: True -> correct evaluation (eval mode + no_grad).
               False -> BUG 5: evaluate in train mode with grads enabled.
    Returns:
        float in [0, 1].
    """
    if fixed:
        # Correct: eval() turns dropout off; no_grad() skips building the
        # autograd graph (saves memory/time). Switch back to train() afterwards.
        model.eval()
        with torch.no_grad():
            logits = model(x)                                        # (N, T, V)
            predictions = logits.argmax(dim=-1)                      # (N, T)
        model.train()
    else:
        # BUG 5: no model.eval() / no torch.no_grad().
        # Dropout is still active -> noisy, pessimistic accuracy,
        # and an autograd graph is built for nothing.
        logits = model(x)                                            # (N, T, V)
        predictions = logits.argmax(dim=-1)                          # (N, T)

    is_correct = (predictions == y).float()                          # (N, T) of 0/1
    return is_correct.mean().item()


def buggy_train(steps=300):
    """Training loop that contains ALL the planted bugs. Returns final accuracy.

    Bugs in here (numbers match the module docstring):
      BUG 1: no optimizer.zero_grad()
      BUG 2: softmax before F.cross_entropy
      BUG 3: labels flattened in transposed order
      BUG 4: inverted causal mask (invert_mask=True)
      BUG 5: evaluation without eval()/no_grad() (fixed=False)
      BUG 6: lr 10x+ too high, and sequential (unshuffled) batches
    """
    x, y = make_data()

    # BUG 4: model built with the inverted causal mask.
    model = TinyAttnModel(invert_mask=True)

    # BUG 6a: lr = 0.1 is far too high for Adam on this model (typical: 1e-3..3e-3).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    for step in range(steps):
        # BUG 6b: batches are taken in a fixed sequential order, never shuffled.
        # (Harmless on this i.i.d. toy data, but in general it biases the gradients
        # if the dataset is sorted, e.g. by length or by label.)
        start = (step * BATCH_SIZE) % 1900
        end = start + BATCH_SIZE
        xb = x[start:end]                                            # (B, T)
        yb = y[start:end]                                            # (B, T)

        # BUG 2: softmax here, and F.cross_entropy applies log_softmax AGAIN.
        # The "logits" become probabilities in [0, 1], so the second softmax is
        # nearly uniform and the gradients are tiny.
        probs = model(xb).softmax(dim=-1)                            # (B, T, V)

        # BUG 3: logits are flattened in (B, T) order, but yb.t() is (T, B),
        # so labels are flattened in (T, B) order -> each prediction is
        # compared against the label of a DIFFERENT token. No error is raised
        # because the total element count still matches.
        flat_logits = probs.view(-1, VOCAB_SIZE)                     # (B*T, V) in (b, t) order
        flat_labels = yb.t().reshape(-1)                             # (B*T,)   in (t, b) order  <-- wrong
        loss = F.cross_entropy(flat_logits, flat_labels)

        # BUG 1: no optimizer.zero_grad() -> .grad keeps summing over all
        # previous steps, so the effective update grows and points in a stale direction.
        loss.backward()
        optimizer.step()

    # BUG 5: evaluate without eval()/no_grad().
    return accuracy(model, x, y, fixed=False)


def fixed_train(steps=300):
    """Correct training loop. Returns final accuracy (should be ~1.0).

    Fixes vs buggy_train:
      FIX 1: optimizer.zero_grad(set_to_none=True) every step
      FIX 2: pass raw logits to F.cross_entropy (it does log_softmax internally)
      FIX 3: flatten logits and labels in the same (B, T) order
      FIX 4: correct lower-triangular causal mask
      FIX 5: evaluate with model.eval() + torch.no_grad()
      FIX 6: lr = 3e-3, random mini-batches; plus gradient clipping for safety
    """
    x, y = make_data()

    # FIX 4: normal causal mask.
    model = TinyAttnModel(invert_mask=False)

    # FIX 6a: sensible Adam learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(steps):
        # FIX 6b: sample a random mini-batch each step (shuffling).
        batch_indices = torch.randint(0, len(x), (BATCH_SIZE,))      # (B,)
        xb = x[batch_indices]                                        # (B, T)
        yb = y[batch_indices]                                        # (B, T)

        # FIX 2: keep the raw logits; cross_entropy does log_softmax itself.
        logits = model(xb)                                           # (B, T, V)

        # FIX 3: flatten both tensors the same way, so row i of logits
        # lines up with element i of labels.
        flat_logits = logits.reshape(-1, VOCAB_SIZE)                 # (B*T, V)
        flat_labels = yb.reshape(-1)                                 # (B*T,)
        loss = F.cross_entropy(flat_logits, flat_labels)

        # FIX 1: clear old gradients BEFORE backward.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Extra safety: clip the global gradient norm to avoid rare spikes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # FIX 5: proper evaluation (dropout off, no autograd graph).
    return accuracy(model, x, y, fixed=True)


if __name__ == "__main__":
    buggy_acc = buggy_train()
    fixed_acc = fixed_train()
    print(f"buggy acc {buggy_acc:.3f}   fixed acc {fixed_acc:.3f}")
    assert fixed_acc > 0.95 and fixed_acc > buggy_acc
    print("OK")
