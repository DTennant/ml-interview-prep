"""Classic ML from scratch: logistic regression (GD), k-means (k-means++ init), PCA via SVD, kNN, conv2d (im2col).

Interview checklist:
- logistic regression grad: X^T (sigmoid(Xw) - y) / N ; use a stable sigmoid / logaddexp for the loss
- k-means: alternate assign / update; handle empty clusters; k-means++ init; objective is non-increasing
- PCA: center X, SVD X = U S V^T, components = V[:k], explained variance = S^2 / (N-1)
- conv2d via im2col: (N,C,H,W) -> patches (N*Ho*Wo, C*kh*kw) @ W.reshape(F, -1).T
"""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid, elementwise. Same shape in and out.

    sigmoid(z) = 1 / (1 + exp(-z))
    For z >= 0 use 1 / (1 + exp(-|z|)); for z < 0 use the equivalent exp(z) / (1 + exp(z)).
    Both forms only ever compute exp(-|z|) <= 1, so nothing overflows.
    """
    exp_neg_abs = np.exp(-np.abs(z))                    # in (0, 1], never overflows
    positive_branch = 1 / (1 + exp_neg_abs)             # correct for z >= 0
    negative_branch = exp_neg_abs / (1 + exp_neg_abs)   # correct for z < 0
    return np.where(z >= 0, positive_branch, negative_branch)


def logreg_fit(X, y, lr=0.5, steps=1000, l2=0.0):
    """Binary logistic regression trained with full-batch gradient descent.

    Input:  X (N, D), y (N,) in {0, 1}
    Output: w (D+1,) weights with the bias as the LAST entry, final loss (scalar)

    Model:  p = sigmoid(Xb @ w), Xb = [X, 1]
    Loss:   BCE = mean(log(1 + exp(z)) - y * z),  z = Xb @ w
    Grad:   Xb^T (p - y) / N  +  l2 * w   (bias is not regularised)
    """
    N = len(X)

    # Step 1: append a column of ones so the bias is just another weight.
    ones_column = np.ones(N)                            # (N,)
    Xb = np.c_[X, ones_column]                          # (N, D+1)
    w = np.zeros(Xb.shape[1])                           # (D+1,)

    for _ in range(steps):
        # Step 2: forward -> logits and probabilities.
        z = Xb @ w                                      # (N,)
        probs = sigmoid(z)                              # (N,)

        # Step 3: gradient of mean BCE w.r.t. w.
        error = probs - y                               # (N,)
        data_grad = Xb.T @ error / N                    # (D+1,)

        # Step 4: L2 penalty on the weights but NOT on the bias (last entry -> 0).
        w_without_bias = np.r_[w[:-1], 0]               # (D+1,)
        reg_grad = l2 * w_without_bias                  # (D+1,)

        # Step 5: gradient descent step.
        grad = data_grad + reg_grad                     # (D+1,)
        w -= lr * grad

    # Stable BCE: log(1 + e^z) = logaddexp(0, z). Uses z from the last iteration.
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss


def kmeans_single(X, k, iters=100, seed=0):
    """One run of k-means (Lloyd's algorithm) with k-means++ initialisation.

    Input:  X (N, D), k clusters
    Output: centers (k, D), assignments (N,), objective = sum of squared distances to assigned center
    """
    rng = np.random.default_rng(seed)
    N = len(X)

    # ---- Step 1: k-means++ initialisation ----
    # First center: a uniformly random point.
    first_index = rng.integers(N)
    centers_list = [X[first_index]]
    # Every next center: sample a point with probability proportional to its squared
    # distance to the nearest center chosen so far -> centers end up spread out.
    for _ in range(1, k):
        current_centers = np.array(centers_list)                        # (j, D)
        diff = X[:, None, :] - current_centers[None, :, :]              # (N, j, D)
        dist_sq_to_each = (diff ** 2).sum(axis=-1)                      # (N, j)
        dist_sq_to_nearest = dist_sq_to_each.min(axis=1)                # (N,)
        sample_probs = dist_sq_to_nearest / dist_sq_to_nearest.sum()    # (N,)
        next_index = rng.choice(N, p=sample_probs)
        centers_list.append(X[next_index])
    centers = np.array(centers_list)                                    # (k, D)

    # ---- Step 2: Lloyd iterations: assign, then update ----
    prev_objective = np.inf
    for _ in range(iters):
        # 2a. Assignment: each point goes to its nearest center.
        diff = X[:, None, :] - centers[None, :, :]                      # (N, k, D)
        dist_sq = (diff ** 2).sum(axis=-1)                              # (N, k)
        assignments = dist_sq.argmin(axis=1)                            # (N,)

        # 2b. Objective (inertia). Must never increase from one iteration to the next.
        objective = dist_sq[np.arange(N), assignments].sum()
        assert objective <= prev_objective + 1e-9

        # 2c. Update: each center moves to the mean of its points.
        for j in range(k):
            members = X[assignments == j]                               # (n_j, D)
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
            else:
                # Empty cluster: re-seed it at a random data point instead of producing NaN.
                centers[j] = X[rng.integers(N)]

        # 2d. Stop when the objective no longer improves.
        if prev_objective - objective < 1e-10:
            break
        prev_objective = objective

    return centers, assignments, objective


def kmeans(X, k, n_init=5, seed=0):
    """k-means only finds a local optimum -> run several inits and keep the lowest objective.

    Output: (centers (k, D), assignments (N,), objective) of the best run.
    """
    best_result = None
    for i in range(n_init):
        result = kmeans_single(X, k, seed=seed + i)
        objective = result[2]
        if best_result is None or objective < best_result[2]:
            best_result = result
    return best_result


def pca(X, k):
    """Principal Component Analysis via SVD.

    Input:  X (N, D), k components to keep
    Output: components (k, D)       top-k directions (rows, unit length)
            projected  (N, k)       data expressed in those directions
            ratio      (k,)         fraction of total variance explained by each component
    """
    # Step 1: center the data (PCA is about variance around the mean).
    mean = X.mean(axis=0)                               # (D,)
    X_centered = X - mean                               # (N, D)

    # Step 2: SVD, X_centered = U S Vt. Rows of Vt are the principal directions,
    # sorted by singular value (largest first).
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)   # U (N, r), S (r,), Vt (r, D)

    # Step 3: keep the top-k directions.
    components = Vt[:k]                                 # (k, D)

    # Step 4: variance along each direction = S^2 / (N - 1).
    variance = S ** 2 / (len(X) - 1)                    # (r,)
    explained_ratio = variance[:k] / variance.sum()     # (k,)

    # Step 5: project the centered data onto the components.
    projected = X_centered @ components.T               # (N, k)
    return components, projected, explained_ratio


def knn_predict(Xtr, ytr, Xte, k=5):
    """k-nearest-neighbour classification by majority vote.

    Input:  Xtr (Ntr, D), ytr (Ntr,) non-negative int labels, Xte (Nte, D)
    Output: predictions (Nte,)
    """
    # Step 1: all pairwise squared distances without a Python loop, using
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    test_sq_norms = (Xte ** 2).sum(axis=1)              # (Nte,)
    train_sq_norms = (Xtr ** 2).sum(axis=1)             # (Ntr,)
    cross_term = Xte @ Xtr.T                            # (Nte, Ntr)
    dist_sq = test_sq_norms[:, None] + train_sq_norms[None, :] - 2 * cross_term   # (Nte, Ntr)

    # Step 2: indices of the k closest training points. argpartition is O(N),
    # cheaper than a full sort; the k neighbours are unordered but that's fine for voting.
    neighbour_idx = np.argpartition(dist_sq, k, axis=1)[:, :k]  # (Nte, k)
    neighbour_labels = ytr[neighbour_idx]                       # (Nte, k)

    # Step 3: majority vote per test point.
    predictions = []
    for labels_of_one_point in neighbour_labels:
        vote_counts = np.bincount(labels_of_one_point)
        predictions.append(vote_counts.argmax())
    return np.array(predictions)


def conv2d_im2col(x, W, b, stride=1, pad=0):
    """2D convolution (cross-correlation, like torch) via im2col + one matrix multiply.

    Input:  x (N, C, H, W_in), W (F, C, kh, kw) filters, b (F,)
    Output: (N, F, Ho, Wo) with Ho = (H + 2*pad - kh) // stride + 1 (same for Wo)

    Idea: copy every receptive-field patch into a row of a big matrix
    (N*Ho*Wo, C*kh*kw), then the convolution is just patches @ filters^T.
    """
    N, C, H, W_in = x.shape
    F, _, kh, kw = W.shape

    # Step 1: zero-pad height and width only.
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))     # (N, C, H+2p, W+2p)

    # Step 2: output spatial size.
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W_in + 2 * pad - kw) // stride + 1

    # Step 3: build the patch tensor. Loop over the kh*kw kernel offsets (small),
    # not over output pixels (large). For offset (i, j), the strided slice picks the
    # input pixel that sits at that offset inside every output position's window.
    cols = np.empty((N, Ho, Wo, C, kh, kw))
    for i in range(kh):
        for j in range(kw):
            row_end = i + stride * Ho
            col_end = j + stride * Wo
            window_pixels = x_padded[:, :, i:row_end:stride, j:col_end:stride]   # (N, C, Ho, Wo)
            cols[:, :, :, :, i, j] = window_pixels.transpose(0, 2, 3, 1)        # (N, Ho, Wo, C)

    # Step 4: flatten patches and filters so the conv becomes one matmul.
    patches = cols.reshape(N * Ho * Wo, C * kh * kw)    # (N*Ho*Wo, C*kh*kw)
    filters = W.reshape(F, C * kh * kw)                 # (F, C*kh*kw)
    out = patches @ filters.T + b                       # (N*Ho*Wo, F)

    # Step 5: reshape back to image layout (N, F, Ho, Wo).
    out = out.reshape(N, Ho, Wo, F)                     # (N, Ho, Wo, F)
    out = out.transpose(0, 3, 1, 2)                     # (N, F, Ho, Wo)
    return out


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
