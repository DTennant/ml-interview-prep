"""Classic ML from scratch: logistic regression (GD), k-means (k-means++ init), PCA via SVD, kNN, conv2d (im2col).

Interview checklist:
- logistic regression grad: X^T (sigmoid(Xw) - y) / N ; use a stable sigmoid / logaddexp for the loss
- k-means: alternate assign / update; handle empty clusters; k-means++ init; objective is non-increasing
- PCA: center X, SVD X = U S V^T, components = V[:k], explained variance = S^2 / (N-1)
- conv2d via im2col: (N,C,H,W) -> patches (N*Ho*Wo, C*kh*kw) @ W.reshape(F, -1).T
"""
import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def logreg_fit(X, y, lr=0.5, steps=1000, l2=0.0):
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        z = Xb @ w
        grad = Xb.T @ (sigmoid(z) - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * grad
    loss = np.mean(np.logaddexp(0, z) - y * z)          # stable BCE
    return w, loss


def kmeans_single(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):                                 # k-means++
        d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    prev = np.inf
    for _ in range(iters):
        d2 = ((X[:, None] - C[None]) ** 2).sum(-1)       # (N, k)
        a = d2.argmin(1)
        obj = d2[np.arange(len(X)), a].sum()
        assert obj <= prev + 1e-9
        for j in range(k):
            C[j] = X[a == j].mean(0) if np.any(a == j) else X[rng.integers(len(X))]
        if prev - obj < 1e-10:
            break
        prev = obj
    return C, a, obj


def kmeans(X, k, n_init=5, seed=0):
    """k-means only finds a local optimum -> run several inits and keep the lowest objective."""
    return min((kmeans_single(X, k, seed=seed + i) for i in range(n_init)), key=lambda r: r[2])


def pca(X, k):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:k]
    var = S ** 2 / (len(X) - 1)
    return comps, (X - mu) @ comps.T, var[:k] / var.sum()


def knn_predict(Xtr, ytr, Xte, k=5):
    # ||a-b||^2 = |a|^2 + |b|^2 - 2ab, vectorised
    d2 = (Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None] - 2 * Xte @ Xtr.T
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]
    votes = ytr[nn_idx]
    return np.array([np.bincount(v).argmax() for v in votes])


def conv2d_im2col(x, W, b, stride=1, pad=0):
    N, C, H, Wd = x.shape
    F, _, kh, kw = W.shape
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kw) // stride + 1
    cols = np.empty((N, Ho, Wo, C, kh, kw))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, :, i, j] = x[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride].transpose(0, 2, 3, 1)
    out = cols.reshape(N * Ho * Wo, -1) @ W.reshape(F, -1).T + b
    return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-2, 1, (200, 2)), rng.normal(2, 1, (200, 2))]
    y = np.r_[np.zeros(200), np.ones(200)]
    w, loss = logreg_fit(X, y)
    acc = ((np.c_[X, np.ones(400)] @ w > 0) == y).mean()
    print(f"logreg loss {loss:.3f} acc {acc:.3f}"); assert acc > 0.95

    blobs = np.r_[rng.normal(0, .3, (100, 2)), rng.normal(3, .3, (100, 2)), rng.normal([0, 3], .3, (100, 2))]
    C, a, obj = kmeans(blobs, 3)
    C = C[np.argsort(C[:, 0] + 10 * C[:, 1])]
    print("kmeans centers\n", np.round(C, 2)); assert np.allclose(C, [[0, 0], [0, 3], [3, 3]], atol=0.15)

    Z = rng.normal(size=(500, 2)) @ np.array([[3, 0, 0], [0, 1, 0]]) + 0.01 * rng.normal(size=(500, 3))
    comps, proj, ratio = pca(Z, 2)
    print("pca explained ratio", np.round(ratio, 3)); assert ratio.sum() > 0.99

    pred = knn_predict(X, y.astype(int), X[::10])
    print("knn acc", (pred == y[::10]).mean())

    import torch
    x = rng.normal(size=(2, 3, 7, 7)); Wc = rng.normal(size=(4, 3, 3, 3)); bc = rng.normal(size=4)
    ref = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(Wc), torch.tensor(bc), stride=2, padding=1).numpy()
    out = conv2d_im2col(x, Wc, bc, stride=2, pad=1)
    print("conv2d max diff vs torch", np.abs(out - ref).max()); assert np.allclose(out, ref)
    print("OK")
