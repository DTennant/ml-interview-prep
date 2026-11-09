"""SGD+momentum, Adam, AdamW from scratch; checked against torch.optim on the same problem.

Interview checklist:
- Adam: m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ; bias-correct mhat = m/(1-b1^t), vhat = v/(1-b2^t)
- update: p -= lr * mhat / (sqrt(vhat) + eps)
- L2 in Adam (add wd*p to g) != AdamW (p -= lr*wd*p separately): with Adam the decay gets rescaled by 1/sqrt(v)
- torch SGD momentum: buf = mu*buf + g ; p -= lr*buf (no dampening by default)
"""
import numpy as np


class SGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.p, self.lr, self.mu, self.wd = params, lr, momentum, weight_decay
        self.buf = [np.zeros_like(x) for x in params]

    def step(self, grads):
        for p, g, b in zip(self.p, grads, self.buf):
            g = g + self.wd * p
            b *= self.mu
            b += g
            p -= self.lr * b


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        self.p, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.wd, self.decoupled, self.t = weight_decay, decoupled, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.decoupled:
                p -= self.lr * self.wd * p            # AdamW
            else:
                g = g + self.wd * p                   # L2 (Adam)
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    return Adam(params, lr, betas, eps, weight_decay, decoupled=True)


def quad_grad(w, A, b):
    """loss = 0.5 w^T A w - b^T w -> grad = A w - b"""
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
