"""LayerNorm / RMSNorm / BatchNorm forward + backward in numpy, checked numerically.

Interview checklist:
- LayerNorm normalises over the feature dim (per token); BatchNorm over the batch dim (per feature)
- BatchNorm keeps running stats for inference; LayerNorm behaves the same in train and eval
- LN backward compact form: dx = (1/sigma) * (g - mean(g) - xhat * mean(g * xhat)), g = dy * gamma
"""
import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm forward: normalise each token over its last (feature) dim.

    Input:  x (..., D), gamma (D,), beta (D,)
    Output: y (..., D), cache for backward

    Formula:
        mu    = mean(x, last dim)
        var   = mean((x - mu)^2, last dim)
        xhat  = (x - mu) / sqrt(var + eps)
        y     = gamma * xhat + beta
    """
    # Step 1: per-token mean and variance over the feature dim.
    mu = x.mean(axis=-1, keepdims=True)                 # (..., 1)
    var = x.var(axis=-1, keepdims=True)                 # (..., 1)

    # Step 2: 1 / std. Kept separately because backward needs it.
    inv_std = 1.0 / np.sqrt(var + eps)                  # (..., 1)

    # Step 3: normalise -> zero mean, unit variance per token.
    x_centered = x - mu                                 # (..., D)
    xhat = x_centered * inv_std                         # (..., D)

    # Step 4: learnable scale and shift.
    y = xhat * gamma + beta                             # (..., D)

    cache = (xhat, inv_std, gamma)
    return y, cache


def layernorm_backward(dy, cache):
    """LayerNorm backward.

    Input:  dy (..., D) upstream gradient, cache from layernorm_forward
    Output: dx (..., D), dgamma (D,), dbeta (D,)

    Formulas:
        dbeta  = sum over all tokens of dy
        dgamma = sum over all tokens of dy * xhat
        g      = dy * gamma                                  (grad w.r.t. xhat)
        dx     = inv_std * (g - mean(g) - xhat * mean(g * xhat))   means over last dim
    The two mean terms come from mu and var both depending on every x_j in the token.
    """
    xhat, inv_std, gamma = cache
    D = xhat.shape[-1]

    # Step 1: parameter gradients. gamma/beta are shared by every token, so we sum
    # over all leading dims (flatten them into one "token" axis first).
    dy_flat = dy.reshape(-1, D)                         # (num_tokens, D)
    xhat_flat = xhat.reshape(-1, D)                     # (num_tokens, D)
    dgamma = (dy_flat * xhat_flat).sum(axis=0)          # (D,)
    dbeta = dy_flat.sum(axis=0)                         # (D,)

    # Step 2: gradient w.r.t. xhat (y = gamma * xhat + beta  ->  dL/dxhat = dy * gamma).
    g = dy * gamma                                      # (..., D)

    # Step 3: gradient through the normalisation.
    # - mean(g) term: removes the part of g that would just shift the mean (from d mu).
    # - xhat * mean(g * xhat) term: removes the part that would change the variance (from d var).
    g_mean = g.mean(axis=-1, keepdims=True)                     # (..., 1)
    g_dot_xhat_mean = (g * xhat).mean(axis=-1, keepdims=True)   # (..., 1)
    dx = inv_std * (g - g_mean - xhat * g_dot_xhat_mean)        # (..., D)

    return dx, dgamma, dbeta


def rmsnorm_forward(x, gamma, eps=1e-6):
    """RMSNorm forward: like LayerNorm but no mean subtraction and no beta.

    Input:  x (..., D), gamma (D,)
    Output: y (..., D), cache

    Formula:
        rms  = sqrt(mean(x^2, last dim) + eps)
        xhat = x / rms
        y    = gamma * xhat
    """
    # Step 1: root-mean-square over the feature dim.
    mean_square = (x * x).mean(axis=-1, keepdims=True)  # (..., 1)
    rms = np.sqrt(mean_square + eps)                    # (..., 1)

    # Step 2: rescale (no centering!).
    xhat = x / rms                                      # (..., D)

    # Step 3: learnable scale.
    y = xhat * gamma                                    # (..., D)

    cache = (xhat, rms, gamma)
    return y, cache


def rmsnorm_backward(dy, cache):
    """RMSNorm backward.

    Input:  dy (..., D), cache from rmsnorm_forward
    Output: dx (..., D), dgamma (D,)

    Formulas:
        dgamma = sum over all tokens of dy * xhat
        g      = dy * gamma
        dx     = (g - xhat * mean(g * xhat)) / rms
    Same as LayerNorm minus the mean(g) term, because there is no mean subtraction.
    """
    xhat, rms, gamma = cache
    D = xhat.shape[-1]

    # Step 1: gamma gradient, summed over every token.
    dy_flat = dy.reshape(-1, D)                         # (num_tokens, D)
    xhat_flat = xhat.reshape(-1, D)                     # (num_tokens, D)
    dgamma = (dy_flat * xhat_flat).sum(axis=0)          # (D,)

    # Step 2: gradient w.r.t. xhat.
    g = dy * gamma                                      # (..., D)

    # Step 3: through xhat = x / rms. rms depends on all x_j, which gives the
    # projection term xhat * mean(g * xhat).
    g_dot_xhat_mean = (g * xhat).mean(axis=-1, keepdims=True)   # (..., 1)
    dx = (g - xhat * g_dot_xhat_mean) / rms                     # (..., D)

    return dx, dgamma


class BatchNorm1d:
    """BatchNorm over a (N, D) batch: normalise each feature across the batch.

    Training: uses the batch mean/var and updates running estimates.
    Eval:     uses the running mean/var (so output doesn't depend on the batch).
    """

    def __init__(self, d, momentum=0.1, eps=1e-5):
        """d = number of features. momentum = weight of the new batch in the running average."""
        # Learnable scale and shift, one per feature.
        self.gamma = np.ones(d)                         # (D,)
        self.beta = np.zeros(d)                         # (D,)

        # Running statistics used at inference time.
        self.running_mean = np.zeros(d)                 # (D,)
        self.running_var = np.ones(d)                   # (D,)

        self.m = momentum
        self.eps = eps
        self.training = True

    def forward(self, x):
        """Forward pass.

        Input:  x (N, D)
        Output: y (N, D) = gamma * (x - mu) / sqrt(var + eps) + beta
        """
        if self.training:
            # Step 1a: statistics of THIS batch, per feature (reduce over axis 0 = batch).
            mu = x.mean(axis=0)                         # (D,)
            var = x.var(axis=0)                         # (D,)  biased (divide by N)
            n = x.shape[0]

            # Step 1b: update running averages for inference.
            # PyTorch stores the UNBIASED variance (divide by N-1) in running_var.
            unbiased_var = var * n / (n - 1)            # (D,)
            self.running_mean = (1 - self.m) * self.running_mean + self.m * mu
            self.running_var = (1 - self.m) * self.running_var + self.m * unbiased_var
        else:
            # Step 1 (eval): use the stored running statistics.
            mu = self.running_mean                      # (D,)
            var = self.running_var                      # (D,)

        # Step 2: normalise.
        inv_std = 1 / np.sqrt(var + self.eps)           # (D,)
        xhat = (x - mu) * inv_std                       # (N, D)

        # Step 3: scale and shift.
        y = xhat * self.gamma + self.beta               # (N, D)

        self.cache = (xhat, inv_std)
        return y

    def backward(self, dy):
        """Backward pass (training mode).

        Input:  dy (N, D)
        Output: dx (N, D); also sets self.dgamma (D,), self.dbeta (D,)

        Same compact formula as LayerNorm, but the means are over the BATCH axis (0):
            g  = dy * gamma
            dx = inv_std * (g - mean_N(g) - xhat * mean_N(g * xhat))
        """
        xhat, inv_std = self.cache

        # Step 1: parameter gradients (sum over the batch).
        self.dgamma = (dy * xhat).sum(axis=0)           # (D,)
        self.dbeta = dy.sum(axis=0)                     # (D,)

        # Step 2: gradient w.r.t. xhat.
        g = dy * self.gamma                             # (N, D)

        # Step 3: through the batch mean and batch variance.
        g_mean = g.mean(axis=0)                         # (D,)
        g_dot_xhat_mean = (g * xhat).mean(axis=0)       # (D,)
        dx = inv_std * (g - g_mean - xhat * g_dot_xhat_mean)    # (N, D)
        return dx


def numgrad(f, x, eps=1e-6):
    """Numerical gradient of a scalar function f() w.r.t. array x (modified in place).

    For every element: grad[i] = (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)
    Output has the same shape as x.
    """
    grad = np.zeros_like(x)
    iterator = np.nditer(x, flags=["multi_index"])
    for _ in iterator:
        i = iterator.multi_index
        original_value = x[i]

        x[i] = original_value + eps
        f_plus = f()

        x[i] = original_value - eps
        f_minus = f()

        x[i] = original_value                           # restore
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5, 8))
    gamma, beta = rng.normal(size=8), rng.normal(size=8)
    dy = rng.normal(size=x.shape)

    y, cache = layernorm_forward(x, gamma, beta)
    dx, dg, db = layernorm_backward(dy, cache)
    f = lambda: (layernorm_forward(x, gamma, beta)[0] * dy).sum()
    for name, a, n in [("dx", dx, numgrad(f, x)), ("dgamma", dg, numgrad(f, gamma)), ("dbeta", db, numgrad(f, beta))]:
        print(f"LayerNorm {name}: max abs err {np.abs(a - n).max():.2e}")
        assert np.allclose(a, n, atol=1e-6)

    y, cache = rmsnorm_forward(x, gamma)
    dx, dg = rmsnorm_backward(dy, cache)
    f = lambda: (rmsnorm_forward(x, gamma)[0] * dy).sum()
    assert np.allclose(dx, numgrad(f, x), atol=1e-6) and np.allclose(dg, numgrad(f, gamma), atol=1e-6)
    print("RMSNorm grads OK")

    bn = BatchNorm1d(8)
    xb, dyb = rng.normal(size=(16, 8)), rng.normal(size=(16, 8))
    bn.forward(xb); dx = bn.backward(dyb)
    f = lambda: (BatchNorm1d(8).forward(xb) * dyb).sum()
    assert np.allclose(dx, numgrad(f, xb), atol=1e-6)
    print("BatchNorm grads OK")
    print("OK")
