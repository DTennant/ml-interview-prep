"""SGD+momentum, Adam, AdamW from scratch; checked against torch.optim on the same problem.

Interview checklist:
- Adam: m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ; bias-correct mhat = m/(1-b1^t), vhat = v/(1-b2^t)
- update: p -= lr * mhat / (sqrt(vhat) + eps)
- L2 in Adam (add wd*p to g) != AdamW (p -= lr*wd*p separately): with Adam the decay gets rescaled by 1/sqrt(v)
- torch SGD momentum: buf = mu*buf + g ; p -= lr*buf (no dampening by default)
"""
import numpy as np


class SGD:
    """SGD with (optional) momentum and L2 weight decay, matching torch.optim.SGD.

    Per parameter p with gradient g:
        g   = g + wd * p          # L2 weight decay folded into the gradient
        buf = mu * buf + g        # momentum buffer (velocity)
        p   = p - lr * buf
    With momentum=0 this is plain gradient descent.
    Parameters are numpy arrays updated IN PLACE (so the caller's arrays change).
    """

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        """params: list of numpy arrays to optimise."""
        self.p = params
        self.lr = lr
        self.mu = momentum
        self.wd = weight_decay

        # One velocity buffer per parameter, same shape, starts at zero.
        self.buf = []
        for param in params:
            self.buf.append(np.zeros_like(param))

    def step(self, grads):
        """Apply one update. grads: list of arrays, same order/shapes as params."""
        for param, grad, velocity in zip(self.p, grads, self.buf):
            # Step 1: L2 weight decay = adding wd * p to the gradient.
            grad_with_decay = grad + self.wd * param

            # Step 2: momentum, buf = mu * buf + g (in place so self.buf stays updated).
            velocity *= self.mu
            velocity += grad_with_decay

            # Step 3: move the parameter against the velocity (in place).
            param -= self.lr * velocity


class Adam:
    """Adam (and AdamW when decoupled=True), matching torch.optim.Adam / AdamW.

    Per parameter p with gradient g, at step t (starting from 1):
        Adam  (L2):        g = g + wd * p
        AdamW (decoupled): p = p - lr * wd * p        (decay applied directly to the weights)
        m    = b1 * m + (1 - b1) * g                  first moment  (mean of g)
        v    = b2 * v + (1 - b2) * g^2                second moment (mean of g^2)
        mhat = m / (1 - b1^t)                         bias correction (m, v start at 0)
        vhat = v / (1 - b2^t)
        p    = p - lr * mhat / (sqrt(vhat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        self.p = params
        self.lr = lr
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.eps = eps
        self.wd = weight_decay
        self.decoupled = decoupled
        self.t = 0                       # number of steps taken so far

        # First and second moment estimates, one per parameter, start at zero.
        self.m = []
        self.v = []
        for param in params:
            self.m.append(np.zeros_like(param))
            self.v.append(np.zeros_like(param))

    def step(self, grads):
        """Apply one Adam/AdamW update to every parameter (in place)."""
        self.t += 1
        bias_correction1 = 1 - self.b1 ** self.t
        bias_correction2 = 1 - self.b2 ** self.t

        for param, grad, m, v in zip(self.p, grads, self.m, self.v):
            # Step 1: weight decay. The two variants differ ONLY here.
            if self.decoupled:
                # AdamW: shrink weights directly; NOT rescaled by 1/sqrt(v).
                param -= self.lr * self.wd * param
            else:
                # Adam + L2: decay goes through the gradient, so it gets rescaled by Adam.
                grad = grad + self.wd * param

            # Step 2: update the running moments (m[:] = ... writes into the stored buffer).
            m[:] = self.b1 * m + (1 - self.b1) * grad
            v[:] = self.b2 * v + (1 - self.b2) * grad * grad

            # Step 3: bias correction. Early on m, v are biased towards 0 because they start at 0.
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Step 4: per-coordinate adaptive step: big where grads are consistent, small where noisy.
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            param -= self.lr * update


def AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    """AdamW = Adam with decoupled weight decay (default wd = 0.01, like torch)."""
    return Adam(params, lr, betas, eps, weight_decay, decoupled=True)


def quad_grad(w, A, b):
    """Gradient of the quadratic loss = 0.5 w^T A w - b^T w  (A symmetric).

    grad = A w - b. Input w (n,), A (n, n), b (n,). Output (n,).
    """
    return A @ w - b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5)); A = M @ M.T + np.eye(5); b = rng.normal(size=5)
    w0 = rng.normal(size=5)

    import torch
    configs = [
        ("SGD+mom", lambda p: SGD(p, 0.01, momentum=0.9, weight_decay=0.01),
         lambda p: torch.optim.SGD(p, 0.01, momentum=0.9, weight_decay=0.01)),
        ("Adam+L2", lambda p: Adam(p, 0.05, weight_decay=0.1),
         lambda p: torch.optim.Adam(p, 0.05, weight_decay=0.1)),
        ("AdamW", lambda p: AdamW(p, 0.05, weight_decay=0.1),
         lambda p: torch.optim.AdamW(p, 0.05, weight_decay=0.1)),
    ]
    for name, mine, ref in configs:
        w = w0.copy(); opt = mine([w])
        wt = torch.tensor(w0, requires_grad=True); topt = ref([wt])
        At, bt = torch.tensor(A), torch.tensor(b)
        for _ in range(200):
            opt.step([quad_grad(w, A, b)])
            topt.zero_grad(); (0.5 * wt @ At @ wt - bt @ wt).backward(); topt.step()
        diff = np.abs(w - wt.detach().numpy()).max()
        print(f"{name:8s} max diff vs torch {diff:.2e}")
        assert diff < 1e-8
    print("OK")
