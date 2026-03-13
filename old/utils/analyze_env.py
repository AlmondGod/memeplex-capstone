import numpy as np
from pettingzoo.mpe import simple_spread_v3

###############################################################################
# 0. Environment basics
###############################################################################
print("=" * 80)
print("SECTION 0: ENVIRONMENT BASICS")
print("=" * 80)

env = simple_spread_v3.env(max_cycles=25, continuous_actions=True)
env.reset(seed=42)

for agent in env.agents:
    obs_space = env.observation_space(agent)
    act_space = env.action_space(agent)
    print(f"\nAgent: {agent}")
    print(f"  Observation space: {obs_space}")
    print(f"  Observation shape: {obs_space.shape}")
    print(f"  Obs low:  {obs_space.low}")
    print(f"  Obs high: {obs_space.high}")
    print(f"  Action space: {act_space}")
    print(f"  Action shape: {act_space.shape}")
    print(f"  Action low:  {act_space.low}")
    print(f"  Action high: {act_space.high}")

env.close()

###############################################################################
# 1. What do the 5 action dimensions mean?
###############################################################################
print("\n" + "=" * 80)
print("SECTION 1: ACTION DIMENSIONS - WHAT DO THEY DO?")
print("=" * 80)

print("""
Per PettingZoo MPE docs for continuous actions in simple_spread:
  Action is a 5-dim vector in [0, 1]:
    dim 0: no action (force magnitude for 'do nothing')
    dim 1: move left  (-x)
    dim 2: move right (+x)
    dim 3: move down  (-y)
    dim 4: move up    (+y)

  The actual applied force = (right - left, up - down) * sensitivity * accel
  The 'no action' dim 0 is effectively ignored for movement.
  Higher values = more force in that direction.
""")

# Empirical test: apply known actions and see position changes
print("Empirical test of action dims:")
for dim_idx in range(5):
    env = simple_spread_v3.env(max_cycles=25, continuous_actions=True)
    env.reset(seed=0)
    
    # Record initial positions
    positions_before = {}
    for agent in env.agent_iter():
        obs, rew, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        positions_before[agent] = obs[:2].copy()  # vel
        action = np.zeros(5, dtype=np.float32)
        action[dim_idx] = 1.0
        env.step(action)
    
    # Next step - read new observations
    positions_after = {}
    for agent in env.agent_iter():
        obs, rew, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        positions_after[agent] = obs[:2].copy()
        env.step(np.zeros(5, dtype=np.float32))  # do nothing
        break  # just need first agent
    
    agent0 = env.possible_agents[0]
    if agent0 in positions_before and agent0 in positions_after:
        delta = positions_after[agent0] - positions_before[agent0]
        print(f"  Action dim {dim_idx} = 1.0 -> agent_0 vel change: dx={delta[0]:.4f}, dy={delta[1]:.4f}")
    env.close()

###############################################################################
# 2. Observation structure
###############################################################################
print("\n" + "=" * 80)
print("SECTION 2: OBSERVATION STRUCTURE (example observations)")
print("=" * 80)

env = simple_spread_v3.env(max_cycles=25, continuous_actions=True)
env.reset(seed=42)

print(f"\nNumber of agents: {len(env.possible_agents)}")
print(f"Possible agents: {env.possible_agents}")

# Try to inspect the underlying world
try:
    par_env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True)
    par_env.reset(seed=42)
    world = par_env.unwrapped.world
    print(f"\nLandmarks: {len(world.landmarks)}")
    for i, lm in enumerate(world.landmarks):
        print(f"  Landmark {i}: pos={lm.state.p_pos}")
    for i, ag in enumerate(world.agents):
        print(f"  Agent {i}: pos={ag.state.p_pos}, vel={ag.state.p_vel}")
    par_env.close()
except Exception as e:
    print(f"Could not inspect world directly: {e}")

step = 0
for agent in env.agent_iter():
    obs, rew, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    if step < 9:  # first 3 steps for each of 3 agents
        print(f"\n  Step {step}, Agent: {agent}")
        print(f"    Obs ({len(obs)} dims): {obs}")
        print(f"    Obs breakdown guess:")
        print(f"      [0:2] self vel:           {obs[0:2]}")
        print(f"      [2:4] self pos:           {obs[2:4]}")
        print(f"      [4:6] landmark_0 rel pos: {obs[4:6]}")
        print(f"      [6:8] landmark_1 rel pos: {obs[6:8]}")
        print(f"      [8:10] landmark_2 rel pos:{obs[8:10]}")
        print(f"      [10:12] other_agent_0 rel:{obs[10:12]}")
        print(f"      [12:14] other_agent_1 rel:{obs[12:14]}")
        if len(obs) > 14:
            print(f"      [14:16] comms_0:          {obs[14:16]}")
            if len(obs) > 16:
                print(f"      [16:18] comms_1:          {obs[16:18]}")
    action = np.zeros(5, dtype=np.float32)
    env.step(action)
    step += 1
env.close()

###############################################################################
# 3. Step-by-step rewards for a single episode
###############################################################################
print("\n" + "=" * 80)
print("SECTION 3: STEP-BY-STEP REWARDS (single episode, random policy)")
print("=" * 80)

env = simple_spread_v3.env(max_cycles=25, continuous_actions=True)
env.reset(seed=42)

step_rewards = {}
step_count = 0
for agent in env.agent_iter():
    obs, rew, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    
    cycle = step_count // 3  # 3 agents
    if cycle not in step_rewards:
        step_rewards[cycle] = {}
    step_rewards[cycle][agent] = rew
    
    action = env.action_space(agent).sample()
    env.step(action)
    step_count += 1

print(f"\nTotal steps taken: {step_count}")
print(f"Total cycles: {len(step_rewards)}")
print(f"\nRewards per cycle (all agents get same reward in cooperative env):")
for cycle_idx in sorted(step_rewards.keys()):
    rews = step_rewards[cycle_idx]
    rew_str = ", ".join(f"{a}: {r:.4f}" for a, r in rews.items())
    print(f"  Cycle {cycle_idx:2d}: {rew_str}")

# Check: are rewards identical across agents?
print("\nAre rewards identical across agents each cycle?")
for cycle_idx in sorted(step_rewards.keys()):
    vals = list(step_rewards[cycle_idx].values())
    if len(set(f"{v:.6f}" for v in vals)) == 1:
        status = "YES (all same)"
    else:
        status = f"NO: {vals}"
    if cycle_idx < 5 or cycle_idx == len(step_rewards) - 1:
        print(f"  Cycle {cycle_idx}: {status}")

env.close()

###############################################################################
# 4. What does the reward actually measure?
###############################################################################
print("\n" + "=" * 80)
print("SECTION 4: REWARD FUNCTION ANALYSIS")
print("=" * 80)

print("""
simple_spread reward (from MPE source):
  reward = -1 * (min distance from each landmark to nearest agent)
           summed over all landmarks
  + collision penalty (if agents overlap, each gets -1 per collision)

So reward is NEGATIVE. Closer to 0 = better.
Perfect score = 0 (each landmark perfectly covered, no collisions).
""")

# Verify with parallel env
par_env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True)
obs, _ = par_env.reset(seed=42)

print("Verifying reward structure with parallel env:")
for step_i in range(5):
    actions = {agent: par_env.action_space(agent).sample() for agent in par_env.agents}
    obs, rewards, terms, truncs, infos = par_env.step(actions)
    
    # Try to compute expected reward from world state
    try:
        world = par_env.unwrapped.world
        dists = []
        for lm in world.landmarks:
            min_d = min(np.linalg.norm(ag.state.p_pos - lm.state.p_pos) for ag in world.agents)
            dists.append(min_d)
        expected_neg_dist = -sum(dists)
        
        actual = list(rewards.values())[0]
        print(f"  Step {step_i}: actual_reward={actual:.4f}, -sum(min_dists)={expected_neg_dist:.4f}, "
              f"diff={actual - expected_neg_dist:.4f} (collision penalty?)")
        for i, lm in enumerate(world.landmarks):
            dists_to_agents = [np.linalg.norm(ag.state.p_pos - lm.state.p_pos) for ag in world.agents]
            print(f"    Landmark {i}: min_dist={min(dists_to_agents):.4f}, all_dists={[f'{d:.4f}' for d in dists_to_agents]}")
    except Exception as e:
        print(f"  Could not inspect world: {e}")
        break

par_env.close()

###############################################################################
# 5. Baseline policies
###############################################################################
def run_policy(policy_fn, policy_name, n_episodes=100, seed_start=0):
    """Run a policy for n episodes, return episode returns."""
    episode_returns = []
    
    for ep in range(n_episodes):
        env = simple_spread_v3.env(max_cycles=25, continuous_actions=True)
        env.reset(seed=seed_start + ep)
        
        ep_reward = 0.0
        agent_count = 0
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            action = policy_fn(obs, agent, env)
            ep_reward += rew
            agent_count += 1
            env.step(action)
        
        # ep_reward is the sum over all agents over all timesteps
        # Since rewards may be shared, divide by num_agents for per-agent return
        n_agents = len(env.possible_agents)
        episode_returns.append(ep_reward / n_agents)
        env.close()
    
    returns = np.array(episode_returns)
    print(f"\n  {policy_name}:")
    print(f"    Mean:   {returns.mean():.4f}")
    print(f"    Std:    {returns.std():.4f}")
    print(f"    Min:    {returns.min():.4f}")
    print(f"    Max:    {returns.max():.4f}")
    print(f"    Median: {np.median(returns):.4f}")
    return returns

print("\n" + "=" * 80)
print("SECTION 5: BASELINE POLICIES (100 episodes each, per-agent episodic return)")
print("=" * 80)

# Random policy
def random_policy(obs, agent, env):
    return env.action_space(agent).sample()

random_returns = run_policy(random_policy, "Random Policy")

# Do nothing policy (action 0 = no-op dim set high)
def do_nothing_policy(obs, agent, env):
    action = np.zeros(5, dtype=np.float32)
    return action

nothing_returns = run_policy(do_nothing_policy, "Do Nothing Policy")

# Go to center policy: observe self position (obs[2:4]), apply force toward (0,0)
def go_to_center_policy(obs, agent, env):
    action = np.zeros(5, dtype=np.float32)
    # obs[2:4] = self position
    pos_x, pos_y = obs[2], obs[3]
    
    # If we're to the right of center, move left (dim 1)
    # If we're to the left, move right (dim 2)
    if pos_x > 0:
        action[1] = min(abs(pos_x), 1.0)  # move left
    else:
        action[2] = min(abs(pos_x), 1.0)  # move right
    
    # If we're above center, move down (dim 3)
    # If below, move up (dim 4)
    if pos_y > 0:
        action[3] = min(abs(pos_y), 1.0)  # move down
    else:
        action[4] = min(abs(pos_y), 1.0)  # move up
    
    return action

center_returns = run_policy(go_to_center_policy, "Go-to-Center Policy")

# Greedy: each agent goes to nearest landmark
def go_to_nearest_landmark_policy(obs, agent, env):
    action = np.zeros(5, dtype=np.float32)
    # obs[4:6], obs[6:8], obs[8:10] = relative positions to landmarks
    landmark_rels = [obs[4:6], obs[6:8], obs[8:10]]
    
    # Find nearest landmark
    dists = [np.linalg.norm(rel) for rel in landmark_rels]
    nearest_idx = np.argmin(dists)
    rel = landmark_rels[nearest_idx]
    
    # Move toward it
    if rel[0] < 0:
        action[1] = min(abs(rel[0]), 1.0)  # move left
    else:
        action[2] = min(abs(rel[0]), 1.0)  # move right
    
    if rel[1] < 0:
        action[3] = min(abs(rel[1]), 1.0)  # move down
    else:
        action[4] = min(abs(rel[1]), 1.0)  # move up
    
    return action

landmark_returns = run_policy(go_to_nearest_landmark_policy, "Go-to-Nearest-Landmark Policy")

###############################################################################
# Summary
###############################################################################
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Environment: simple_spread_v3, max_cycles=25, continuous_actions=True
  3 agents, 3 landmarks
  Obs dim: 18, Action dim: 5 (continuous [0,1])
  Reward: negative sum of min-distances from landmarks to agents (+ collision penalty)
  Perfect score: 0.0 (all landmarks covered, no collisions)

Per-agent episode return (mean over 100 episodes):
  Random:                {random_returns.mean():.4f} +/- {random_returns.std():.4f}
  Do Nothing:            {nothing_returns.mean():.4f} +/- {nothing_returns.std():.4f}
  Go to Center:          {center_returns.mean():.4f} +/- {center_returns.std():.4f}
  Go to Nearest Landmark:{landmark_returns.mean():.4f} +/- {landmark_returns.std():.4f}

Key insight: Since landmarks are randomly placed and the episode is short (25 steps),
even simple heuristic policies that move toward landmarks do noticeably better.
A well-trained RL agent should achieve returns close to 0 (maybe -20 to -50 range
is "good", while near 0 is "excellent").
""")
