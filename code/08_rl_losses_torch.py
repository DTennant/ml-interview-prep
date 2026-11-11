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
    """REINFORCE (vanilla policy gradient) loss with a baseline.

    Formula:
        A_i  = R_i - b
        loss = -mean_i( log pi(a_i) * A_i )
    Minimising this loss = gradient ascent on expected return, because
    grad E[R] = E[ grad log pi(a) * (R - b) ].

    Args:
        logp:     (N,) log pi(a_i) of the sampled actions (requires grad).
        returns:  (N,) return R_i observed for each sample.
        baseline: optional scalar / (N,) tensor b. Defaults to mean(returns).
                  Any baseline that does not depend on a_i keeps the gradient
                  unbiased but reduces its variance.
    Returns:
        scalar loss tensor.
    """
    # Step 1: pick the baseline (batch mean if none given).
    if baseline is None:
        baseline_value = returns.mean()
    else:
        baseline_value = baseline

    # Step 2: advantage = how much better than "average" this sample was.
    advantage = returns - baseline_value                              # (N,)

    # Step 3: detach the advantage: it is a fixed weight, gradients must only
    # flow through log pi, never through the reward/baseline.
    advantage = advantage.detach()                                    # (N,)

    # Step 4: negative because optimisers minimise.
    weighted_logp = logp * advantage                                  # (N,)
    loss = -weighted_logp.mean()
    return loss


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation, computed backwards in time.

    Formulas (nonterm_t = 1 - done_t):
        delta_t = r_t + gamma * V(s_{t+1}) * nonterm_t - V(s_t)        (TD error)
        A_t     = delta_t + gamma * lam * nonterm_t * A_{t+1}
        R_t     = A_t + V(s_t)                                         (value target)
    lam=0 -> one-step TD (low variance, high bias); lam=1 -> Monte Carlo (opposite).

    Args:
        rewards: (T,) reward at each step.
        values:  (T+1,) value estimates V(s_0..s_T); the last entry is the
                 bootstrap value of the state after the final step.
        dones:   (T,) 1.0 if the episode ended at step t, else 0.0.
    Returns:
        advantages: (T,)
        returns:    (T,) = advantages + values[:-1]
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros(num_steps)                               # (T,)

    # A_{T} = 0 beyond the end of the rollout.
    next_advantage = 0.0

    # Walk backwards so A_{t+1} is already known when computing A_t.
    for t in reversed(range(num_steps)):
        # Step 1: if the episode ended at t, do not bootstrap from s_{t+1}
        # and do not carry advantage across the episode boundary.
        not_terminal = 1.0 - dones[t]

        # Step 2: one-step TD error.
        bootstrapped_next_value = gamma * values[t + 1] * not_terminal
        td_error = rewards[t] + bootstrapped_next_value - values[t]

        # Step 3: exponentially-weighted sum of future TD errors.
        advantage_t = td_error + gamma * lam * not_terminal * next_advantage
        advantages[t] = advantage_t
        next_advantage = advantage_t

    # Value-function regression target.
    returns = advantages + values[:-1]                                # (T,)
    return advantages, returns


def ppo_loss(logp_new, logp_old, adv, mask, clip_eps=0.2):
    """PPO clipped surrogate loss, averaged over valid (response) tokens.

    Formula:
        ratio = exp(logp_new - logp_old)       = pi_new / pi_old
        L_tok = -min( ratio * A, clip(ratio, 1-eps, 1+eps) * A )
        loss  = sum(L_tok * mask) / sum(mask)  (token-level mean)
    Taking the min makes the objective pessimistic: once the ratio moves
    outside [1-eps, 1+eps] in the direction that helps, the gradient is zero,
    so the new policy cannot drift too far from the old one in one update.

    Args:
        logp_new: (B, T) log-probs under the current policy (requires grad).
        logp_old: (B, T) log-probs under the policy that generated the data.
        adv:      (B, T) advantages (per token, or a sequence value broadcast).
        mask:     (B, T) 1 for response tokens, 0 for prompt / padding.
        clip_eps: clip range epsilon.
    Returns:
        scalar loss.
    """
    # Step 1: probability ratio pi_new / pi_old (computed in log space for stability).
    log_ratio = logp_new - logp_old                                   # (B, T)
    ratio = torch.exp(log_ratio)                                      # (B, T)

    # Step 2: the two surrogate objectives.
    unclipped_objective = ratio * adv                                 # (B, T)
    clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)           # (B, T)
    clipped_objective = clipped_ratio * adv                           # (B, T)

    # Step 3: pessimistic (minimum) objective, negated to get a loss.
    per_token_loss = -torch.min(unclipped_objective, clipped_objective)  # (B, T)

    # Step 4: average over valid tokens only.
    masked_loss_sum = (per_token_loss * mask).sum()
    num_valid_tokens = mask.sum()
    return masked_loss_sum / num_valid_tokens


def grpo_advantages(rewards, group_size, eps=1e-6):
    """GRPO group-relative advantages (no value network needed).

    For each prompt we sampled G = group_size responses. Each response's
    advantage is its reward standardised within its own group:
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Args:
        rewards:    (B*G,) rewards, ordered so the G responses of the same
                    prompt are contiguous: [p0_r0..p0_r{G-1}, p1_r0, ...].
        group_size: G.
        eps:        avoids division by zero when every response in a group
                    got the same reward (then all advantages are 0).
    Returns:
        (B*G,) advantages in the same order.
    """
    # Step 1: one row per prompt.
    rewards_per_group = rewards.view(-1, group_size)                  # (B, G)

    # Step 2: per-group statistics.
    group_mean = rewards_per_group.mean(dim=1, keepdim=True)          # (B, 1)
    group_std = rewards_per_group.std(dim=1, keepdim=True)            # (B, 1)  (unbiased std)

    # Step 3: standardise within each group.
    centered = rewards_per_group - group_mean                         # (B, G)
    advantages = centered / (group_std + eps)                         # (B, G)

    # Step 4: back to the original flat layout.
    return advantages.view(-1)                                        # (B*G,)


def k3_kl(logp, logp_ref):
    """Per-sample "k3" estimator of KL(pi || pi_ref), using samples from pi.

    Formula, with d = log pi_ref(x) - log pi(x) = log(ref/pi):
        k3 = exp(d) - d - 1
    Properties:
        - unbiased: E_pi[exp(d)] = 1, so E[k3] = E[-d] = KL(pi || pi_ref)
        - always >= 0 (since e^d >= 1 + d), unlike the naive estimator -d
        - lower variance than -d
    Used in GRPO as the per-token KL penalty to the reference model.

    Args:
        logp:     (...) log pi(x) for samples x ~ pi.
        logp_ref: (...) log pi_ref(x) for the same samples.
    Returns:
        (...) per-sample KL estimates (take .mean() for the KL estimate).
    """
    log_ratio_ref_over_policy = logp_ref - logp                       # (...)
    k3 = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1
    return k3


def token_logprobs(logits, tokens):
    """Log-probability of each chosen token.

    Args:
        logits: (B, T, V) model outputs.
        tokens: (B, T) token ids actually generated / observed.
    Returns:
        (B, T) log pi(tokens[b, t]).
    """
    # Step 1: normalise over the vocabulary.
    log_probs_all_vocab = torch.log_softmax(logits, dim=-1)           # (B, T, V)

    # Step 2: pick out the log-prob of the token that was actually used.
    # gather needs an index with the same number of dims -> add a trailing dim.
    token_index = tokens.unsqueeze(-1)                                # (B, T, 1)
    chosen_log_probs = log_probs_all_vocab.gather(-1, token_index)    # (B, T, 1)
    return chosen_log_probs.squeeze(-1)                               # (B, T)


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- 1. REINFORCE on a 5-armed bandit: arm 2 has the highest mean reward ---
    true_means = torch.tensor([0.1, 0.2, 0.9, 0.3, 0.0])
    policy_logits = torch.zeros(5, requires_grad=True)
    optimizer = torch.optim.Adam([policy_logits], lr=0.05)
    for _ in range(500):
        policy = torch.distributions.Categorical(logits=policy_logits)
        actions = policy.sample((64,))                                # (64,)
        noise = 0.1 * torch.randn(64)
        rewards = true_means[actions] + noise                         # (64,)
        loss = reinforce_loss(policy.log_prob(actions), rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    action_probs = policy_logits.softmax(0)
    print("bandit policy", action_probs.detach().numpy().round(3))
    assert action_probs.argmax() == 2 and action_probs[2] > 0.9

    # --- 2. GAE sanity: lam=1, gamma=1, V=0 -> advantage = reward-to-go ---
    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.zeros(4)
    dones = torch.tensor([0.0, 0.0, 1.0])
    advantages, returns = gae(rewards, values, dones, gamma=1.0, lam=1.0)
    print("GAE", advantages.tolist())
    assert torch.allclose(advantages, torch.tensor([3.0, 2.0, 2.0]))

    # --- 3. PPO: ratio = 1 -> loss = -mean(adv); clipping kicks in for large ratio ---
    logp_same = torch.zeros(2, 3)
    adv = torch.tensor([[1.0, -1.0, 2.0], [0.5, 0.5, 0.0]])
    mask = torch.ones(2, 3)
    assert torch.isclose(ppo_loss(logp_same, logp_same, adv, mask), -(adv.mean()))
    logp_shifted = torch.full((2, 3), 1.0)                            # ratio = e ~ 2.72 > 1.2
    print("PPO clipped loss", ppo_loss(logp_shifted, logp_same, adv, mask).item())

    # --- 4. GRPO: advantages are zero-mean within each group ---
    group_rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    grpo_adv = grpo_advantages(group_rewards, 4)
    print("GRPO adv", grpo_adv.numpy().round(3))
    assert torch.allclose(grpo_adv.view(2, 4).mean(1), torch.zeros(2), atol=1e-6)

    # --- 5. k3 KL is >= 0 and ~ true KL on average ---
    policy_dist = torch.distributions.Categorical(logits=torch.randn(10))
    reference_dist = torch.distributions.Categorical(logits=torch.randn(10))
    samples = policy_dist.sample((200000,))
    k3_estimate = k3_kl(policy_dist.log_prob(samples), reference_dist.log_prob(samples)).mean()
    true_kl = torch.distributions.kl_divergence(policy_dist, reference_dist)
    print(f"k3 KL est {k3_estimate:.4f} true {true_kl:.4f}")
    assert abs(k3_estimate - true_kl) < 0.02
    print("OK")
