"""LayerNorm / RMSNorm / BatchNorm forward + backward in numpy, checked numerically.

Interview checklist:
- LayerNorm normalises over the feature dim (per token); BatchNorm over the batch dim (per feature)
- BatchNorm keeps running stats for inference; LayerNorm behaves the same in train and eval
- LN backward compact form: dx = (1/sigma) * (g - mean(g) - xhat * mean(g * xhat)), g = dy * gamma
"""
import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * gamma + beta, (xhat, inv, gamma)


def layernorm_backward(dy, cache):
    xhat, inv, gamma = cache
    dgamma = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    dbeta = dy.reshape(-1, dy.shape[-1]).sum(0)
    g = dy * gamma
    dx = inv * (g - g.mean(-1, keepdims=True) - xhat * (g * xhat).mean(-1, keepdims=True))
    return dx, dgamma, dbeta


def rmsnorm_forward(x, gamma, eps=1e-6):
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    xhat = x / rms
    return xhat * gamma, (xhat, rms, gamma)


def rmsnorm_backward(dy, cache):
    xhat, rms, gamma = cache
    dgamma = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    g = dy * gamma
    dx = (g - xhat * (g * xhat).mean(-1, keepdims=True)) / rms
    return dx, dgamma


class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)
        self.running_mean, self.running_var = np.zeros(d), np.ones(d)
        self.m, self.eps, self.training = momentum, eps, True

    def forward(self, x):                      # x: (N, D)
        if self.training:
            mu, var = x.mean(0), x.var(0)
            n = x.shape[0]
            self.running_mean = (1 - self.m) * self.running_mean + self.m * mu
            self.running_var = (1 - self.m) * self.running_var + self.m * var * n / (n - 1)  # unbiased
        else:
            mu, var = self.running_mean, self.running_var
        inv = 1 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        self.cache = (xhat, inv)
        return xhat * self.gamma + self.beta

    def backward(self, dy):
        xhat, inv = self.cache
        self.dgamma = (dy * xhat).sum(0)
        self.dbeta = dy.sum(0)
        g = dy * self.gamma
        return inv * (g - g.mean(0) - xhat * (g * xhat).mean(0))


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


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
