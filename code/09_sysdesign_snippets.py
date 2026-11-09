"""Small coding pieces that tend to show up inside a system design round.

1. Async worker pool with timeout + retry + exponential backoff (rollout / eval runner)
2. UCB1 bandit for allocating budget across candidate ideas
3. Paired bootstrap / significance test: is candidate B really better than A?
4. Content-hash cache + dedup for experiment results
5. Successive halving (budgeted search over candidates)
"""
import asyncio
import hashlib
import json
import math
import random


# ---------- 1. worker pool ----------
async def run_with_retry(fn, item, timeout, retries=3, base_delay=0.05):
    for attempt in range(retries + 1):
        try:
            return {"item": item, "ok": True, "result": await asyncio.wait_for(fn(item), timeout)}
        except Exception as e:  # noqa: BLE001  (TimeoutError included)
            if attempt == retries:
                return {"item": item, "ok": False, "error": type(e).__name__}
            await asyncio.sleep(base_delay * 2 ** attempt * (1 + random.random()))   # backoff + jitter


async def worker_pool(fn, items, concurrency=8, timeout=1.0, retries=3):
    sem = asyncio.Semaphore(concurrency)

    async def guarded(it):
        async with sem:
            return await run_with_retry(fn, it, timeout, retries)

    return await asyncio.gather(*(guarded(i) for i in items))


# ---------- 2. UCB1 ----------
class UCB1:
    def __init__(self, n_arms, c=2.0):
        self.n, self.s, self.c, self.t = [0] * n_arms, [0.0] * n_arms, c, 0

    def select(self):
        self.t += 1
        for i, n in enumerate(self.n):
            if n == 0:
                return i
        return max(range(len(self.n)),
                   key=lambda i: self.s[i] / self.n[i] + math.sqrt(self.c * math.log(self.t) / self.n[i]))

    def update(self, i, reward):
        self.n[i] += 1
        self.s[i] += reward


# ---------- 3. paired bootstrap ----------
def paired_bootstrap(scores_a, scores_b, n_boot=5000, seed=0):
    """Per-task scores of A and B on the same tasks. Returns (mean diff, 95% CI, p(B<=A))."""
    rng = random.Random(seed)
    d = [b - a for a, b in zip(scores_a, scores_b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    p = sum(m <= 0 for m in means) / n_boot
    return sum(d) / n, (means[int(.025 * n_boot)], means[int(.975 * n_boot)]), p


# ---------- 4. cache ----------
class ResultCache:
    def __init__(self):
        self.store = {}

    @staticmethod
    def key(code: str, config: dict, seed: int) -> str:
        blob = json.dumps({"code": code, "config": config, "seed": seed}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get_or_run(self, code, config, seed, run):
        k = self.key(code, config, seed)
        if k not in self.store:
            self.store[k] = run()
        return self.store[k]


# ---------- 5. successive halving ----------
def successive_halving(candidates, evaluate, budget0=1, eta=2):
    """evaluate(candidate, budget) -> score (higher better)."""
    pool, budget = list(candidates), budget0
    while len(pool) > 1:
        scored = sorted(pool, key=lambda c: evaluate(c, budget), reverse=True)
        pool, budget = scored[:max(1, len(pool) // eta)], budget * eta
    return pool[0]


if __name__ == "__main__":
    random.seed(0)

    async def flaky(x):
        await asyncio.sleep(random.random() * 0.02)
        if random.random() < 0.3:
            raise RuntimeError("transient")
        if x == 13:
            await asyncio.sleep(10)           # always times out
        return x * x

    res = asyncio.run(worker_pool(flaky, range(20), concurrency=5, timeout=0.2, retries=3))
    ok = [r for r in res if r["ok"]]
    print(f"worker pool: {len(ok)}/20 ok, failed: {[r['item'] for r in res if not r['ok']]}")
    assert all(r["ok"] for r in res if r["item"] != 13) and not res[13]["ok"]

    probs = [0.2, 0.5, 0.8, 0.4]
    b = UCB1(4)
    for _ in range(3000):
        i = b.select(); b.update(i, float(random.random() < probs[i]))
    print("UCB pulls", b.n); assert max(range(4), key=lambda i: b.n[i]) == 2

    a = [random.gauss(0.5, 0.1) for _ in range(100)]
    bb = [x + 0.03 + random.gauss(0, 0.05) for x in a]
    diff, ci, p = paired_bootstrap(a, bb)
    print(f"bootstrap diff {diff:.3f} CI ({ci[0]:.3f},{ci[1]:.3f}) p={p:.4f}"); assert ci[0] > 0

    c = ResultCache(); calls = []
    for _ in range(3):
        c.get_or_run("print(1)", {"lr": 1e-3}, 0, lambda: calls.append(1) or 0.9)
    assert len(calls) == 1; print("cache OK (1 run for 3 requests)")

    true_q = {f"c{i}": random.random() for i in range(16)}
    best = successive_halving(true_q, lambda cand, bud: true_q[cand] + random.gauss(0, 0.3 / math.sqrt(bud)))
    print("successive halving picked", best, "true best", max(true_q, key=true_q.get))
    print("OK")
