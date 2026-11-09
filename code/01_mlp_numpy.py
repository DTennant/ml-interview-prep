"""MLP from scratch in numpy: forward, backward, softmax-CE, gradient check, training.

Interview checklist:
- say shapes out loud: X (N, D), W1 (D, H), b1 (H,), W2 (H, C), b2 (C,)
- softmax must subtract the row max (log-sum-exp trick)
- dL/dlogits = (softmax - onehot) / N
- verify with a centered finite-difference gradient check
"""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def cross_entropy(logits, y):
    """logits (N, C), y (N,) int labels -> scalar mean loss"""
    N = logits.shape[0]
    return -log_softmax(logits)[np.arange(N), y].mean()


class MLP:
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hidden)),
            "b1": np.zeros(d_hidden),
            "W2": rng.normal(0, np.sqrt(2.0 / d_hidden), (d_hidden, d_out)),
            "b2": np.zeros(d_out),
        }

    def forward(self, X):
        p = self.params
        h_pre = X @ p["W1"] + p["b1"]       # (N, H)
        h = np.maximum(h_pre, 0)            # ReLU
        logits = h @ p["W2"] + p["b2"]      # (N, C)
        self.cache = (X, h_pre, h)
        return logits

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        loss = cross_entropy(logits, y)
        X, h_pre, h = self.cache
        N = X.shape[0]
        p = self.params

        dlogits = softmax(logits)
        dlogits[np.arange(N), y] -= 1
        dlogits /= N                                  # (N, C)

        grads = {}
        grads["W2"] = h.T @ dlogits                   # (H, C)
        grads["b2"] = dlogits.sum(0)                  # (C,)
        dh = dlogits @ p["W2"].T                      # (N, H)
        dh_pre = dh * (h_pre > 0)                     # ReLU backward
        grads["W1"] = X.T @ dh_pre                    # (D, H)
        grads["b1"] = dh_pre.sum(0)
        return loss, grads


def grad_check(model, X, y, eps=1e-5, n_samples=20, seed=0):
    """Centered finite differences on random coordinates. Returns max relative error."""
    rng = np.random.default_rng(seed)
    _, grads = model.loss_and_grads(X, y)
    worst = 0.0
    for name, W in model.params.items():
        for _ in range(n_samples):
            idx = tuple(rng.integers(0, s) for s in W.shape)
            old = W[idx]
            W[idx] = old + eps
            lp, _ = model.loss_and_grads(X, y)
            W[idx] = old - eps
            lm, _ = model.loss_and_grads(X, y)
            W[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


def make_spirals(n=300, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n, c))
    return np.vstack(X), np.concatenate(y)


if __name__ == "__main__":
    X, y = make_spirals()
    model = MLP(2, 64, 3)
    # float64 + a random batch. Avoid points sitting exactly on the ReLU kink (e.g. X=0 with b1=0):
    # finite differences across the kink disagree with the analytic subgradient.
    idx = np.random.default_rng(1).choice(len(X), 16, replace=False)
    idx = idx[np.abs(X[idx]).sum(1) > 1e-3]
    err = grad_check(model, X[idx], y[idx])
    print(f"grad check max rel error: {err:.2e}")
    assert err < 1e-6

    lr = 1.0
    for step in range(2000):
        loss, grads = model.loss_and_grads(X, y)
        for k in model.params:
            model.params[k] -= lr * grads[k]
        if step % 500 == 0:
            print(f"step {step:4d} loss {loss:.4f}")
    acc = (model.forward(X).argmax(1) == y).mean()
    print(f"final loss {loss:.4f}, train acc {acc:.3f}")
    assert acc > 0.9
    print("OK")
