"""MLP from scratch in numpy: forward, backward, softmax-CE, gradient check, training.

Interview checklist:
- say shapes out loud: X (N, D), W1 (D, H), b1 (H,), W2 (H, C), b2 (C,)
- softmax must subtract the row max (log-sum-exp trick)
- dL/dlogits = (softmax - onehot) / N
- verify with a centered finite-difference gradient check
"""
import numpy as np


def softmax(z):
    """Row-wise softmax.

    Input:  z (N, C) raw scores (logits)
    Output: probs (N, C), every row is non-negative and sums to 1

    Formula: softmax(z)_j = exp(z_j) / sum_k exp(z_k)
    """
    # Step 1: subtract the max of each row. Softmax is shift-invariant, so the
    # result is unchanged, but now the largest exponent is exp(0) = 1 -> no overflow.
    row_max = z.max(axis=1, keepdims=True)              # (N, 1)
    z_shifted = z - row_max                             # (N, C)

    # Step 2: exponentiate.
    exp_z = np.exp(z_shifted)                           # (N, C)

    # Step 3: normalise each row so it sums to 1.
    row_sum = exp_z.sum(axis=1, keepdims=True)          # (N, 1)
    probs = exp_z / row_sum                             # (N, C)
    return probs


def log_softmax(z):
    """Row-wise log-softmax, computed stably.

    Input:  z (N, C)
    Output: log_probs (N, C)

    Formula: log_softmax(z)_j = z_j - log(sum_k exp(z_k))
    Computing log(softmax(z)) directly can give log(0) = -inf, so we use the
    log-sum-exp form instead.
    """
    # Step 1: shift by the row max (same reason as in softmax).
    row_max = z.max(axis=1, keepdims=True)              # (N, 1)
    z_shifted = z - row_max                             # (N, C)

    # Step 2: log of the normaliser, log(sum_k exp(z_k)).
    sum_exp = np.exp(z_shifted).sum(axis=1, keepdims=True)  # (N, 1)
    log_normaliser = np.log(sum_exp)                    # (N, 1)

    # Step 3: subtract it.
    log_probs = z_shifted - log_normaliser              # (N, C)
    return log_probs


def cross_entropy(logits, y):
    """Mean cross-entropy loss.

    Input:  logits (N, C), y (N,) integer class labels in [0, C)
    Output: scalar loss = -(1/N) * sum_i log p(y_i | x_i)
    """
    N = logits.shape[0]
    log_probs = log_softmax(logits)                     # (N, C)

    # Pick out the log-probability of the correct class for each sample.
    row_indices = np.arange(N)                          # (N,)
    log_prob_correct = log_probs[row_indices, y]        # (N,)

    loss = -log_prob_correct.mean()                     # scalar
    return loss


class MLP:
    """Two-layer MLP: Linear -> ReLU -> Linear.

    Shapes:
        X  (N, D)  input batch
        W1 (D, H), b1 (H,)   first layer
        W2 (H, C), b2 (C,)   second layer
        logits (N, C)
    """

    def __init__(self, d_in, d_hidden, d_out, seed=0):
        """Create parameters.

        Weights use He initialisation (std = sqrt(2 / fan_in)), which keeps the
        activation variance roughly constant through ReLU layers.
        Biases start at zero.
        """
        rng = np.random.default_rng(seed)

        std_W1 = np.sqrt(2.0 / d_in)
        W1 = rng.normal(0, std_W1, (d_in, d_hidden))    # (D, H)
        b1 = np.zeros(d_hidden)                         # (H,)

        std_W2 = np.sqrt(2.0 / d_hidden)
        W2 = rng.normal(0, std_W2, (d_hidden, d_out))   # (H, C)
        b2 = np.zeros(d_out)                            # (C,)

        self.params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
        }

    def forward(self, X):
        """Forward pass.

        Input:  X (N, D)
        Output: logits (N, C)

        Also stores the intermediate values needed by the backward pass in self.cache.
        """
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        # Step 1: first linear layer.
        h_pre = X @ W1 + b1                             # (N, H)

        # Step 2: ReLU non-linearity, max(0, x).
        h = np.maximum(h_pre, 0)                        # (N, H)

        # Step 3: second linear layer gives the class scores.
        logits = h @ W2 + b2                            # (N, C)

        # Save what backward needs: X for dW1, h_pre for the ReLU mask, h for dW2.
        self.cache = (X, h_pre, h)
        return logits

    def loss_and_grads(self, X, y):
        """Compute the loss and the gradient of the loss w.r.t. every parameter.

        Input:  X (N, D), y (N,) int labels
        Output: (loss scalar, grads dict with the same keys/shapes as self.params)

        Backward formulas (chain rule, from the loss back to the input):
            dlogits = (softmax(logits) - onehot(y)) / N        (N, C)
            dW2     = h^T @ dlogits                            (H, C)
            db2     = sum_over_batch(dlogits)                  (C,)
            dh      = dlogits @ W2^T                           (N, H)
            dh_pre  = dh * 1[h_pre > 0]                        (N, H)
            dW1     = X^T @ dh_pre                             (D, H)
            db1     = sum_over_batch(dh_pre)                   (H,)
        """
        # ---- forward ----
        logits = self.forward(X)                        # (N, C)
        loss = cross_entropy(logits, y)                 # scalar
        X, h_pre, h = self.cache
        N = X.shape[0]
        W2 = self.params["W2"]

        # ---- backward ----
        # Step 1: gradient of mean cross-entropy w.r.t. logits.
        # d(-log softmax_y)/dz = softmax(z) - onehot(y); the 1/N comes from the mean.
        probs = softmax(logits)                         # (N, C)
        dlogits = probs.copy()                          # (N, C)
        dlogits[np.arange(N), y] -= 1                   # subtract the one-hot target
        dlogits = dlogits / N                           # (N, C)

        # Step 2: second linear layer, logits = h @ W2 + b2.
        # dL/dW2 = h^T @ dL/dlogits
        dW2 = h.T @ dlogits                             # (H, C)
        # b2 is added to every row, so its gradient is the sum over rows.
        db2 = dlogits.sum(axis=0)                       # (C,)
        # dL/dh = dL/dlogits @ W2^T  (pass gradient to the previous layer)
        dh = dlogits @ W2.T                             # (N, H)

        # Step 3: ReLU, h = max(0, h_pre).
        # Derivative is 1 where h_pre > 0, else 0 -> gradient only flows through active units.
        relu_mask = (h_pre > 0)                         # (N, H) bool
        dh_pre = dh * relu_mask                         # (N, H)

        # Step 4: first linear layer, h_pre = X @ W1 + b1.
        dW1 = X.T @ dh_pre                              # (D, H)
        db1 = dh_pre.sum(axis=0)                        # (H,)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return loss, grads


def grad_check(model, X, y, eps=1e-5, n_samples=20, seed=0):
    """Compare analytic gradients with centered finite differences.

    For n_samples random coordinates of each parameter:
        numeric = (L(w + eps) - L(w - eps)) / (2 * eps)
        rel_err = |numeric - analytic| / (|numeric| + |analytic|)

    Returns the worst (max) relative error seen. Should be ~1e-7 or smaller in float64.
    """
    rng = np.random.default_rng(seed)
    _, analytic_grads = model.loss_and_grads(X, y)
    worst_rel_error = 0.0

    for name, W in model.params.items():
        for _ in range(n_samples):
            # Step 1: choose a random coordinate inside this parameter tensor.
            idx = tuple(rng.integers(0, dim_size) for dim_size in W.shape)
            original_value = W[idx]

            # Step 2: loss with the coordinate nudged up.
            W[idx] = original_value + eps
            loss_plus, _ = model.loss_and_grads(X, y)

            # Step 3: loss with the coordinate nudged down.
            W[idx] = original_value - eps
            loss_minus, _ = model.loss_and_grads(X, y)

            # Step 4: restore the parameter (very important!).
            W[idx] = original_value

            # Step 5: compare numeric slope with the analytic gradient.
            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            analytic_grad = analytic_grads[name][idx]
            denominator = max(1e-8, abs(numeric_grad) + abs(analytic_grad))
            rel_error = abs(numeric_grad - analytic_grad) / denominator

            worst_rel_error = max(worst_rel_error, rel_error)

    return worst_rel_error


def make_spirals(n=300, k=3, seed=0):
    """Toy dataset: k interleaved 2D spirals with n points each.

    Output: X (n*k, 2) float, y (n*k,) int labels in [0, k)
    """
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    for class_id in range(k):
        radius = np.linspace(0.0, 1, n)                                 # (n,)
        angle_start = class_id * 4
        angle_end = (class_id + 1) * 4
        noise = rng.normal(0, 0.2, n)                                   # (n,)
        angle = np.linspace(angle_start, angle_end, n) + noise          # (n,)

        x_coord = radius * np.sin(angle)                                # (n,)
        y_coord = radius * np.cos(angle)                                # (n,)
        points = np.c_[x_coord, y_coord]                                # (n, 2)

        X_parts.append(points)
        y_parts.append(np.full(n, class_id))

    X = np.vstack(X_parts)                                              # (n*k, 2)
    y = np.concatenate(y_parts)                                         # (n*k,)
    return X, y


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
