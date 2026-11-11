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
    """Call `await fn(item)` with a per-attempt timeout, retrying on any failure.

    Retry schedule (exponential backoff with jitter):
        sleep_k = base_delay * 2**k * (1 + U[0,1))      for attempt k = 0, 1, ...
    Jitter spreads retries out so many failing workers don't all hit the
    backend again at the same moment ("thundering herd").

    Args:
        fn:         async function taking one item.
        item:       the work item.
        timeout:    seconds allowed per attempt (asyncio.TimeoutError on expiry).
        retries:    number of extra attempts after the first (total = retries + 1).
        base_delay: first backoff delay in seconds.
    Returns:
        {"item", "ok": True, "result"} on success, or
        {"item", "ok": False, "error": <exception class name>} after the last failure.
        Never raises: one bad item must not crash the whole pool.
    """
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        try:
            # Step 1: run one attempt, cancelled if it exceeds `timeout`.
            result = await asyncio.wait_for(fn(item), timeout)
            return {"item": item, "ok": True, "result": result}
        except Exception as error:  # noqa: BLE001  (TimeoutError included)
            # Step 2: out of attempts -> report the failure instead of raising.
            is_last_attempt = attempt == retries
            if is_last_attempt:
                return {"item": item, "ok": False, "error": type(error).__name__}

            # Step 3: back off before retrying: doubles each attempt, plus jitter.
            exponential_delay = base_delay * 2 ** attempt
            jitter_factor = 1 + random.random()
            await asyncio.sleep(exponential_delay * jitter_factor)


async def worker_pool(fn, items, concurrency=8, timeout=1.0, retries=3):
    """Process all `items` with at most `concurrency` calls in flight at once.

    Args:
        fn:          async function applied to each item.
        items:       iterable of work items.
        concurrency: max simultaneous calls (protects rate limits / GPUs).
        timeout, retries: forwarded to run_with_retry.
    Returns:
        list of result dicts (see run_with_retry), in the SAME order as items.
    """
    # A semaphore with N permits: each task must hold one while running.
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one_with_limit(item):
        """Wait for a free slot, then run the item with retries."""
        async with semaphore:
            return await run_with_retry(fn, item, timeout, retries)

    # Create all tasks up front; the semaphore throttles them.
    # gather() preserves the input order in its output.
    tasks = [run_one_with_limit(item) for item in items]
    return await asyncio.gather(*tasks)


# ---------- 2. UCB1 ----------
class UCB1:
    """UCB1 multi-armed bandit.

    Score of arm i at time t:
        UCB_i = mean_reward_i + sqrt( c * ln(t) / n_i )
    The first term exploits arms that look good; the second term (the
    "uncertainty bonus") is big for rarely-pulled arms, forcing exploration.
    Every arm is pulled once before the formula is used (n_i = 0 would divide by zero).

    Attributes:
        n: list[int]   number of pulls per arm.
        s: list[float] sum of rewards per arm.
        c: exploration constant (2.0 = classic UCB1).
        t: total number of select() calls so far.
    """

    def __init__(self, n_arms, c=2.0):
        """Start with zero pulls and zero reward on every arm."""
        self.n = [0] * n_arms
        self.s = [0.0] * n_arms
        self.c = c
        self.t = 0

    def _ucb_score(self, arm):
        """Mean reward of `arm` plus its exploration bonus (requires n[arm] > 0)."""
        mean_reward = self.s[arm] / self.n[arm]
        exploration_bonus = math.sqrt(self.c * math.log(self.t) / self.n[arm])
        return mean_reward + exploration_bonus

    def select(self):
        """Return the index of the arm to pull next."""
        self.t += 1

        # Step 1: make sure every arm has been tried at least once.
        for arm, pull_count in enumerate(self.n):
            if pull_count == 0:
                return arm

        # Step 2: otherwise pick the arm with the highest UCB score.
        best_arm = 0
        best_score = -math.inf
        for arm in range(len(self.n)):
            score = self._ucb_score(arm)
            if score > best_score:
                best_arm = arm
                best_score = score
        return best_arm

    def update(self, i, reward):
        """Record that arm `i` was pulled and returned `reward`."""
        self.n[i] += 1
        self.s[i] += reward


# ---------- 3. paired bootstrap ----------
def paired_bootstrap(scores_a, scores_b, n_boot=5000, seed=0):
    """Paired bootstrap test: is system B really better than A on the same tasks?

    Because both systems ran on the SAME tasks, we work with per-task
    differences d_j = b_j - a_j (pairing removes task-difficulty noise).

    Procedure:
        1. d = [b_j - a_j]
        2. repeat n_boot times: resample n tasks WITH replacement, record mean(d*)
        3. 95% CI = 2.5th and 97.5th percentiles of those bootstrap means
        4. p-value (one-sided) ~ fraction of bootstrap means <= 0

    Args:
        scores_a, scores_b: per-task scores, same length and task order.
        n_boot: number of bootstrap resamples.
        seed:   RNG seed for reproducibility.
    Returns:
        (mean_diff, (ci_low, ci_high), p_value_B_not_better)
    """
    rng = random.Random(seed)

    # Step 1: per-task differences.
    differences = []
    for score_a, score_b in zip(scores_a, scores_b):
        differences.append(score_b - score_a)
    num_tasks = len(differences)

    # Step 2: bootstrap distribution of the mean difference.
    bootstrap_means = []
    for _ in range(n_boot):
        resample_total = 0.0
        for _ in range(num_tasks):
            random_task = rng.randrange(num_tasks)
            resample_total += differences[random_task]
        bootstrap_means.append(resample_total / num_tasks)
    bootstrap_means.sort()

    # Step 3: how often does the resampled mean say "B is not better"?
    num_not_better = 0
    for mean_value in bootstrap_means:
        if mean_value <= 0:
            num_not_better += 1
    p_value = num_not_better / n_boot

    # Step 4: percentile confidence interval.
    ci_low = bootstrap_means[int(0.025 * n_boot)]
    ci_high = bootstrap_means[int(0.975 * n_boot)]

    observed_mean_diff = sum(differences) / num_tasks
    return observed_mean_diff, (ci_low, ci_high), p_value


# ---------- 4. cache ----------
class ResultCache:
    """In-memory cache of experiment results keyed by a content hash.

    The key is sha256 over (code, config, seed), so identical experiments are
    deduplicated and run only once, regardless of who submits them.
    """

    def __init__(self):
        """Create an empty key -> result store."""
        self.store = {}

    @staticmethod
    def key(code: str, config: dict, seed: int) -> str:
        """Deterministic hex hash of everything that determines the result.

        sort_keys=True makes {"a":1,"b":2} and {"b":2,"a":1} hash identically.
        """
        payload = {"code": code, "config": config, "seed": seed}
        canonical_json = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def get_or_run(self, code, config, seed, run):
        """Return the cached result, or call `run()` once and cache what it returns."""
        cache_key = self.key(code, config, seed)
        if cache_key not in self.store:
            # Cache miss: run the experiment (expensive) exactly once.
            self.store[cache_key] = run()
        return self.store[cache_key]


# ---------- 5. successive halving ----------
def successive_halving(candidates, evaluate, budget0=1, eta=2):
    """Budgeted search: evaluate everyone cheaply, keep the best 1/eta, give survivors eta x more budget.

    Round r: |pool| = N / eta**r candidates, each evaluated with budget0 * eta**r.
    Bad candidates are dropped early on cheap (noisy) evaluations; the
    expensive, accurate evaluations are spent only on the few finalists.

    Args:
        candidates: iterable of candidates.
        evaluate:   evaluate(candidate, budget) -> score (higher is better).
                    Larger budget should mean less noisy score.
        budget0:    budget per candidate in the first round.
        eta:        keep the top 1/eta each round and multiply the budget by eta.
    Returns:
        the single surviving candidate.
    """
    pool = list(candidates)
    budget = budget0

    while len(pool) > 1:
        # Step 1: score every remaining candidate at the current budget
        # (each candidate is evaluated exactly once per round).
        ranked = sorted(pool, key=lambda candidate: evaluate(candidate, budget), reverse=True)

        # Step 2: keep the top 1/eta (at least one survivor).
        num_survivors = max(1, len(pool) // eta)
        pool = ranked[:num_survivors]

        # Step 3: survivors get a bigger budget next round.
        budget = budget * eta

    return pool[0]


if __name__ == "__main__":
    random.seed(0)

    # --- 1. worker pool: flaky jobs recover via retries; item 13 always times out ---
    async def flaky(x):
        """Fails transiently 30% of the time; item 13 hangs forever."""
        await asyncio.sleep(random.random() * 0.02)
        if random.random() < 0.3:
            raise RuntimeError("transient")
        if x == 13:
            await asyncio.sleep(10)           # always times out
        return x * x

    results = asyncio.run(worker_pool(flaky, range(20), concurrency=5, timeout=0.2, retries=3))
    succeeded = [r for r in results if r["ok"]]
    failed_items = [r["item"] for r in results if not r["ok"]]
    print(f"worker pool: {len(succeeded)}/20 ok, failed: {failed_items}")
    assert all(r["ok"] for r in results if r["item"] != 13) and not results[13]["ok"]

    # --- 2. UCB1: should pull the best arm (index 2, p=0.8) most often ---
    arm_probs = [0.2, 0.5, 0.8, 0.4]
    bandit = UCB1(4)
    for _ in range(3000):
        arm = bandit.select()
        reward = float(random.random() < arm_probs[arm])
        bandit.update(arm, reward)
    print("UCB pulls", bandit.n)
    assert max(range(4), key=lambda i: bandit.n[i]) == 2

    # --- 3. paired bootstrap: B is A + 0.03 on average -> CI should exclude 0 ---
    scores_a = [random.gauss(0.5, 0.1) for _ in range(100)]
    scores_b = [x + 0.03 + random.gauss(0, 0.05) for x in scores_a]
    diff, ci, p = paired_bootstrap(scores_a, scores_b)
    print(f"bootstrap diff {diff:.3f} CI ({ci[0]:.3f},{ci[1]:.3f}) p={p:.4f}")
    assert ci[0] > 0

    # --- 4. cache: three identical requests -> only one real run ---
    cache = ResultCache()
    run_calls = []

    def expensive_run():
        """Pretend experiment that records each time it is actually executed."""
        run_calls.append(1)
        return 0.9

    for _ in range(3):
        cache.get_or_run("print(1)", {"lr": 1e-3}, 0, expensive_run)
    assert len(run_calls) == 1
    print("cache OK (1 run for 3 requests)")

    # --- 5. successive halving: noise shrinks as 1/sqrt(budget) ---
    true_quality = {f"c{i}": random.random() for i in range(16)}

    def noisy_evaluate(candidate, budget):
        """True quality plus Gaussian noise whose std shrinks with budget."""
        return true_quality[candidate] + random.gauss(0, 0.3 / math.sqrt(budget))

    best = successive_halving(true_quality, noisy_evaluate)
    print("successive halving picked", best, "true best", max(true_quality, key=true_quality.get))
    print("OK")
