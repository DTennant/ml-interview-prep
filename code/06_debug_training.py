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


def make_data(n=2000, T=8, V=10):
    # task: predict the previous token (copy shifted by 1) -> needs attention to the past only
    x = torch.randint(1, V, (n, T))
    y = torch.cat([torch.zeros(n, 1, dtype=torch.long), x[:, :-1]], 1)
    return x, y


class TinyAttnModel(nn.Module):
    def __init__(self, V=10, d=32, T=8, invert_mask=False):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, V)
        self.drop = nn.Dropout(0.3)
        self.invert = invert_mask

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T))
        q, k, v = self.qkv(h).chunk(3, -1)
        att = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        if self.invert:
            mask = ~mask                                     # BUG 4
        att = att.masked_fill(~mask, -1e9).softmax(-1)
        return self.out(self.drop(att @ v) + h)


def accuracy(model, x, y, fixed):
    if fixed:
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(-1)                       # (B, T)
        model.train()
    else:
        pred = model(x).argmax(-1)                           # BUG 5: no model.eval() / no_grad (dropout on)
    return (pred == y).float().mean().item()


def buggy_train(steps=300):
    x, y = make_data()
    model = TinyAttnModel(invert_mask=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)      # BUG 6a: lr too high
    for step in range(steps):
        i = (step * 64) % 1900
        xb, yb = x[i:i + 64], y[i:i + 64]                    # BUG 6b: no shuffle (fine here, bad in general)
        logits = model(xb).softmax(-1)                       # BUG 2
        loss = F.cross_entropy(logits.view(-1, 10), yb.t().reshape(-1))   # BUG 3
        loss.backward()                                      # BUG 1: no zero_grad
        opt.step()
    return accuracy(model, x, y, fixed=False)


def fixed_train(steps=300):
    x, y = make_data()
    model = TinyAttnModel(invert_mask=False)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        idx = torch.randint(0, len(x), (64,))
        xb, yb = x[idx], y[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 10), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return accuracy(model, x, y, fixed=True)


if __name__ == "__main__":
    b = buggy_train()
    f = fixed_train()
    print(f"buggy acc {b:.3f}   fixed acc {f:.3f}")
    assert f > 0.95 and f > b
    print("OK")
