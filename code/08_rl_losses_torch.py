"""RL for LLM agents: REINFORCE with baseline, GAE, PPO clipped loss, GRPO group advantage, k3 KL estimator.
Plus a runnable sanity check: REINFORCE learns a 5-armed bandit.

Interview checklist:
- policy gradient: grad E[R] = E[grad log pi(a|s) * A]; baseline reduces variance, keeps it unbiased
- GAE: delta_t = r_t + gamma V(s_{t+1}) (1-done) - V(s_t); A_t = delta_t + gamma*lam*(1-done) A_{t+1}
- PPO: ratio = exp(logp_new - logp_old); L = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
- GRPO: sample G responses per prompt; A_i = (r_i - mean(r)) / (std(r) + eps); no value net
- token-level masking: only response tokens count; average over valid tokens (be explicit about seq vs token mean)
- KL to reference: k3 = exp(ref - logp) - (ref - logp) - 1  (unbiased, always >= 0)
"""
import torch


def reinforce_loss(logp, returns, baseline=None):
    adv = returns - (returns.mean() if baseline is None else baseline)
    return -(logp * adv.detach()).mean()


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """rewards, dones: (T,), values: (T+1,) incl. bootstrap value. Returns advantages (T,), returns (T,)."""
    T = rewards.shape[0]
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values[:-1]


def ppo_loss(logp_new, logp_old, adv, mask, clip_eps=0.2):
    """all (B, T); mask = 1 for response tokens."""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    per_tok = -torch.min(unclipped, clipped)
    return (per_tok * mask).sum() / mask.sum()


def grpo_advantages(rewards, group_size, eps=1e-6):
    """rewards (B*G,) ordered by prompt -> advantages (B*G,)."""
    r = rewards.view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def k3_kl(logp, logp_ref):
    d = logp_ref - logp
    return torch.exp(d) - d - 1


def token_logprobs(logits, tokens):
    """logits (B, T, V), tokens (B, T) -> log pi(token) (B, T)"""
    return torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    # --- REINFORCE on a bandit ---
    true_means = torch.tensor([0.1, 0.2, 0.9, 0.3, 0.0])
    logits = torch.zeros(5, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(500):
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample((64,))
        r = true_means[a] + 0.1 * torch.randn(64)
        loss = reinforce_loss(dist.log_prob(a), r)
        opt.zero_grad(); loss.backward(); opt.step()
    p = logits.softmax(0)
    print("bandit policy", p.detach().numpy().round(3)); assert p.argmax() == 2 and p[2] > 0.9

    # --- GAE sanity: lam=1, gamma=1, V=0 -> advantage = reward-to-go ---
    r = torch.tensor([1.0, 0.0, 2.0]); v = torch.zeros(4); d = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = gae(r, v, d, gamma=1.0, lam=1.0)
    print("GAE", adv.tolist()); assert torch.allclose(adv, torch.tensor([3.0, 2.0, 2.0]))

    # --- PPO: ratio = 1 -> loss = -mean(adv) ; clipping kicks in for large ratio ---
    lp = torch.zeros(2, 3); A = torch.tensor([[1.0, -1.0, 2.0], [0.5, 0.5, 0.0]]); m = torch.ones(2, 3)
    assert torch.isclose(ppo_loss(lp, lp, A, m), -(A.mean()))
    big = torch.full((2, 3), 1.0)  # ratio = e
    print("PPO clipped loss", ppo_loss(big, lp, A, m).item())

    # --- GRPO ---
    adv = grpo_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]), 4)
    print("GRPO adv", adv.numpy().round(3)); assert torch.allclose(adv.view(2, 4).mean(1), torch.zeros(2), atol=1e-6)

    # --- k3 KL is >= 0 and ~ true KL on average ---
    q = torch.distributions.Categorical(logits=torch.randn(10)); pr = torch.distributions.Categorical(logits=torch.randn(10))
    s = q.sample((200000,))
    est = k3_kl(q.log_prob(s), pr.log_prob(s)).mean()
    true = torch.distributions.kl_divergence(q, pr)
    print(f"k3 KL est {est:.4f} true {true:.4f}"); assert abs(est - true) < 0.02
    print("OK")
