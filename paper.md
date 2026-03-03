# Joint Discovery of Observation and Reward Functions via LLM-Guided Evolution

**Anonymous Authors**

**Keywords:** reinforcement learning, reward shaping, observation design, large language models, evolutionary search, quality-diversity

## Abstract

When an RL agent fails to learn, is the reward function to blame—or the observation space? Recent LLM-guided approaches assume the answer is always the reward and evolve only reward functions, leaving observations fixed. We show this assumption is often wrong: on a relational reasoning task, evolving observations alone achieves 99.4% success while evolving rewards alone achieves 2.2%. More importantly, on complex multi-phase tasks, *neither component alone suffices*—joint observation-reward evolution reaches 78.6% where the best single-component ablation reaches only 30.6%. We formalize this as MDP interface discovery and present an environment-agnostic system that evolves both components as executable code through LLM-guided MAP-Elites search. The system requires only a natural language task description and an API reference—no hand-designed reward or observation. Across three grid-world tasks and two continuous-control tasks (Go1 quadruped, Panda arm), the same codebase discovers working interfaces at $3–11 per run. Analysis of evolved solutions reveals that joint optimization discovers tightly coupled phase-conditioned observations and phase-gated rewards—a coordination that cannot emerge when components are evolved independently.

---

## 1. Introduction

Consider the task of picking up an object, carrying it across a multi-room grid, and placing it next to a target. A standard RL environment provides a flattened grid observation and a sparse completion reward. An agent trained with this default interface achieves 0.2% success. What went wrong?

The conventional diagnosis would focus on reward sparsity, and indeed, recent LLM-guided methods like Eureka (Ma et al., 2024) and Text2Reward (Xie et al., 2024) have demonstrated impressive results by evolving dense reward functions. But evolving a sophisticated multi-phase reward for our task—while keeping the default observation—yields only 5.0% success. The problem is not just the reward. It is the *observation*: the flattened grid cannot efficiently encode the spatial relationships (distances to objects, adjacency to targets, valid placement locations) that any reasonable reward function would need to reference. The observation and reward are not independent design choices—they must agree on task structure.

This paper studies the joint design of observations and rewards, which we term the *MDP interface*. We make three observations that motivate this work:

**Observation design is often the primary bottleneck, not reward design.** On a medium-difficulty relational task (placing an object adjacent to a target), evolving only the observation achieves 99.4% success with the environment's built-in sparse reward. Evolving only the reward achieves 2.2%. The 45× gap suggests that the Eureka paradigm—evolving rewards atop fixed observations—addresses the wrong bottleneck in many settings.

**Joint optimization discovers coordination that independent optimization cannot.** On the hard multi-room task, the best jointly-evolved interface uses 147-dimensional observations with explicit phase encodings (`phase0`: seek object, `phase1`: carry to target, `phase2`: place) and a reward function with matching phase-gated shaping. This observation-reward coupling—where the observation tells the agent *what phase it's in* and the reward tells it *what to do in that phase*—cannot emerge when the two components are evolved separately.

**The same system works across fundamentally different domains.** By isolating all environment-specific code behind an adapter interface, the same evolution codebase handles discrete grid worlds (XLand-MiniGrid, JAX) and continuous physics simulation (MuJoCo/Brax, JAX) with no core modifications. This suggests MDP interface discovery is a general capability, not a domain-specific trick.

We formalize MDP interface discovery as bilevel optimization (Section 2) and present a system combining LLM-guided code generation, three-stage crash filtering, cascade evaluation, and MAP-Elites with island-model diversity (Section 3). We evaluate on three novel grid-world tasks of increasing difficulty and two continuous-control tasks (Section 4), with 10-seed statistical evaluation on grid worlds. Our main contributions:

1. We provide empirical evidence that joint observation-reward optimization is necessary for complex tasks, achieving 78.6% success where the best single-component ablation achieves 30.6% (10 seeds, $p < 0.01$).

2. We show that observation design—not reward design—is the primary bottleneck in multiple settings, directly challenging the assumptions underlying Eureka-style approaches.

3. We demonstrate environment-agnostic MDP interface discovery across discrete and continuous domains using a single system, at $3–11 per evolution run.

---

## 2. Problem Formulation

### 2.1 The MDP Interface

An RL environment exposes a raw simulator state $s \in \mathcal{S}_{\text{raw}}$ (grid contents, joint angles, sensor readings). The agent, however, interacts through an *interface*: it receives observation $o = \phi(s)$ and optimizes shaped reward $r = R(s, a, s')$. We define the **MDP interface** as the pair $\mathcal{I} = (\phi, R)$.

The interface determines the effective POMDP the agent faces. A poorly chosen $\phi$ can make an otherwise-solvable task intractable by discarding critical state information or presenting it in a form the policy network cannot extract. A poorly chosen $R$ can provide no learning signal for intermediate progress. When both are poor, the agent has no chance.

### 2.2 Interface Discovery as Bilevel Optimization

We seek the interface that maximizes task success when a standard RL algorithm is trained with it:

$$\mathcal{I}^* = \arg\max_{(\phi, R)} \; \mathbb{E}\left[ \text{Success}\!\left(\pi^*_{\phi, R}\right) \right]$$

where $\pi^*_{\phi, R}$ is the converged policy from training PPO with observation mapping $\phi$ and reward $R$. This is bilevel optimization: the outer loop searches over interfaces (via LLM-guided evolution), and the inner loop trains policies (via PPO). The interface is represented as executable code—two Python functions—generated and refined by an LLM.

The system requires three human-provided inputs: (1) a natural language task description, (2) a condensed API reference describing the environment's state structure (~2 pages), and (3) a training configuration file. No hand-designed observation or reward is required.

### 2.3 Decomposing the Interface: Ablation Modes

To isolate the contribution of each component, we define ablation modes that fix one component while evolving the other:

| Mode | Observation $\phi$ | Reward $R$ | Analogue |
|------|-------------------|------------|----------|
| **Full** | Evolved | Evolved | Our approach |
| **Obs-only** | Evolved | Env built-in | Tests obs contribution |
| **Reward-only** | Env default | Evolved | Eureka paradigm |
| **Sparse** | Env default | Env built-in | Lower bound |

The reward-only mode is functionally equivalent to Eureka (Ma et al., 2024): it evolves reward functions while keeping observations fixed at the environment default. Our comparison between full and reward-only directly measures the value of co-evolving observations.

---

## 3. Method

### 3.1 Overview

The system operates as an evolutionary loop. Each iteration: (1) the LLM generates candidate interface code conditioned on the current population and training feedback, (2) candidates pass through a three-stage crash filter that catches syntax errors, import failures, and shape mismatches before any training begins, (3) surviving candidates undergo cascade evaluation—short training followed by full multi-seed training for promising candidates, and (4) evaluated candidates enter a MAP-Elites archive that maintains structural diversity across reward complexity and observation dimensionality.

```
Task Description ──┐
Context Document ──┤    ┌─────────┐    ┌──────────┐    ┌──────────┐
Config YAML ───────┴──> │   LLM   │──> │  Crash   │──> │ Cascade  │
                        │ Generate │    │  Filter  │    │  Train   │
                        └────▲────┘    │ (3-stage)│    │ (PPO)    │
                             │         └──────────┘    └────┬─────┘
                             │                              │
                             │    ┌────────────────┐        │
                             └────│   MAP-Elites   │<───────┘
                                  │ (Island Model) │
                                  └────────────────┘
```

### 3.2 Environment Adapter

All environment-specific code is isolated behind an abstract adapter interface. The adapter provides: environment construction with injected obs/reward functions, dummy states for crash-filter validation, success computation, and default obs/reward functions for ablation modes. To support a new domain, one implements this interface and writes a context document—the core system (evolution controller, LLM prompting, crash filter, evaluator) requires no changes. Our XLand-MiniGrid and MuJoCo/Brax adapters share zero code beyond this interface.

### 3.3 LLM-Guided Code Generation

The LLM receives a structured prompt containing: (1) a generic system prompt with RL engineering guidelines and no environment-specific vocabulary, (2) the context document describing state structure and action space, (3) the task description, and (4) evolutionary context—parent code from the archive, training metrics from prior evaluations, and error messages from crashed candidates. The prompt is mode-aware: in reward-only mode, observation instructions are omitted.

The LLM generates complete Python functions that must be JAX-traceable (`jax.jit` compatible), use only `jax` and `jax.numpy`, and conform to prescribed signatures: `get_observation(state) -> array` and `compute_reward(state, action, next_state) -> scalar`.

### 3.4 Crash Filter and Cascade Evaluation

LLM-generated code frequently contains errors—30–50% of candidates fail in practice. A three-stage filter catches these before training: Stage 0 verifies AST structure and function signatures (5s); Stage 1 loads the module and catches import errors (10s); Stage 2 executes functions on a real environment state and verifies output shapes (30–60s). Error messages feed back to the LLM for self-correction.

Candidates surviving the crash filter enter cascade evaluation: short training (0.5–6M steps) assesses viability, and only candidates exceeding a 5% success threshold proceed to full multi-seed training (1–30M steps, 3 seeds). This filters the ~70% of non-crashing candidates that still produce degenerate rewards or uninformative observations, avoiding expensive full evaluations.

### 3.5 MAP-Elites with Island Model

To prevent premature convergence, we maintain diversity using MAP-Elites (Mouret & Clune, 2015) with two behavioral dimensions: reward function complexity (AST node count) and observation dimensionality. The feature space is discretized into a 5×5 grid; each cell stores the highest-fitness interface with those characteristics. This ensures simple compact solutions coexist with complex ones.

We augment MAP-Elites with an island model: 2–3 independent populations with ring-topology migration every 5–20 generations at 10–20% rate. Parent selection mixes exploitation (70% fitness-proportional), exploration (20% uniform random), and elite selection (10% top performers).

---

## 4. Experimental Setup

### 4.1 Environments

We evaluate on five tasks across two domains. All tasks are novel—not drawn from standard benchmarks—to ensure no prior LLM exposure.

**XLand-MiniGrid** (discrete, JAX-compiled):
- **Easy**: Pick up blue pyramid. 9×9 grid, 80 steps. Tests basic navigation.
- **Medium**: Place yellow pyramid adjacent to green square. 9×9 grid, 80 steps. Tests relational reasoning.
- **Hard**: Pick up blue pyramid (transforms to green ball on pickup), place green ball next to yellow hex. 13×13 four-room grid with doors and distractors, 400 steps. Tests multi-phase planning with object transformation.

**MuJoCo/Brax** (continuous, JAX-compiled):
- **Go1 Push Recovery**: Unitree Go1 quadruped recovers from random force impulses (50–250N). Success: mean position error < 10cm.
- **Panda Tracking**: Franka Panda arm tracks a 3D Lissajous trajectory. Success: mean tracking error < 2cm.

### 4.2 Evaluation Protocol

XLand-MiniGrid: **10 seeds** per mode-task combination. Each seed runs 30 evolution iterations and reports the best interface's success rate over 80 evaluation episodes. We report mean $\pm$ standard deviation.

MuJoCo: 30 evolution iterations with 3-seed cascade evaluation per candidate.

### 4.3 Implementation

LLM: Claude Sonnet 4.6 (AWS Bedrock), temperature 0.7. RL: PPO with GRU (grid worlds) or MLP (continuous control), 2048–8192 parallel environments. 30 evolution iterations per run. Full hyperparameters in Appendix C.

---

## 5. Results

### 5.1 Joint Optimization is Necessary for Complex Tasks

**Table 1: Success rate (%) on XLand-MiniGrid tasks (mean $\pm$ std, 10 seeds).**

| Task | Full (Ours) | Obs-Only | Reward-Only | Sparse |
|------|:-----------:|:--------:|:-----------:|:------:|
| Easy | **96.8 $\pm$ 2.9** | 89.8 $\pm$ 29.9 | 75.8 $\pm$ 6.4 | 28.0 $\pm$ 6.7 |
| Medium | 94.4 $\pm$ 11.2 | **99.4 $\pm$ 1.8** | 2.2 $\pm$ 1.9 | 5.8 $\pm$ 3.5 |
| Hard | **78.6 $\pm$ 4.9** | 30.6 $\pm$ 28.8 | 5.0 $\pm$ 6.6 | 0.2 $\pm$ 0.6 |

The critical result is the hard task. Joint evolution achieves 78.6%, outperforming obs-only (30.6%) by 2.6× and reward-only (5.0%) by 15.7×. Neither single-component ablation approaches the joint result—confirming that observation and reward must be co-designed for multi-phase tasks.

On the medium task, obs-only *exceeds* full (99.4% vs. 94.4%), demonstrating that when the built-in reward is adequate, co-evolving the reward adds variance without benefit. This is not a failure of joint optimization but a useful diagnostic: it reveals which component is the bottleneck.

### 5.2 Observation Design is the Primary Bottleneck

The most striking result in Table 1 is the medium task: obs-only achieves 99.4% while reward-only achieves 2.2%—a 45× gap. The environment's sparse reward ("did you place the object adjacent to the target?") is sufficient signal when the observation encodes the right relational features. Conversely, even sophisticated evolved rewards cannot compensate for the default observation (a flattened tile grid) that lacks explicit distance-to-target and adjacency information.

This pattern holds directionally across tasks. On easy, obs-only (89.8%) outperforms reward-only (75.8%). On hard, the gap narrows (30.6% vs. 5.0%) because the sparse reward provides almost no signal for sequential multi-phase behavior. But the consistent obs-only advantage challenges the assumption underlying Eureka and similar systems: that environments provide adequate observations and only rewards need design.

### 5.3 How Joint Optimization Achieves Coordination

Why does joint evolution succeed where independent evolution fails on the hard task? Examining the best evolved interface (147 dimensions, 78.6%) reveals tight observation-reward coupling:

The observation includes explicit **phase encodings**: `phase0` (seeking pyramid), `phase1` (carrying to hex), `phase2` (placed successfully). It provides a **dynamic target** that switches based on phase: `target = select(holding, hex_pos, pyramid_pos)`. It computes **placement-relevant features**: for each cardinal neighbor of the hex, whether it's a floor tile (valid placement), its distance from the agent, and whether the agent is currently there.

The reward function uses **matching phase gates**: pyramid approach shaping (weight 3.0) activates only in phase 0, hex approach and placement shaping (weights 3.0 and 1.5) activate only in phase 1. Milestone bonuses (+5 for pickup, +20 for correct placement, -5 for wrong placement) mark phase transitions.

This coupling is the key insight: the observation encodes phase information *because* the reward needs it for gating, and the reward uses phase gating *because* the observation makes phase information available. This co-adaptation cannot emerge when evolving components independently—obs-only has no reward to align with, and reward-only has no phase encoding to condition on.

In contrast, the obs-only best interface (472 dimensions, 30.6%) provides exhaustive spatial features but with the fixed sparse reward, the agent receives no intermediate guidance. The reward-only best interface (5.0%) has phase-gated shaping, but the default observation cannot efficiently represent the spatial relationships the reward references.

### 5.4 Evolutionary Search vs. Single-Shot Generation

On the easy task, evolutionary search achieves 99% success compared to 12% for single-shot LLM generation (identical prompt, no feedback loop). The 8.25× improvement confirms that iterative refinement with training metrics is essential—LLMs cannot reliably produce working MDP interfaces in one attempt.

### 5.5 Transfer to Continuous Control

The same system transfers to MuJoCo continuous control with only an adapter change:

**Table 2: Continuous control results.**

| Task | Success | Obs Dim | LLM Cost | Wall Time |
|------|:-------:|:-------:|:--------:|:---------:|
| Go1 Push Recovery | 32% | 61 | $3.05 | 5.4 hrs |
| Panda Tracking | *running* | — | — | — |

The Go1 interface is qualitatively different from grid-world solutions: it discovers body-frame IMU projections, direction-to-origin rotated by heading, tanh-normalized velocities, and upright-gated position rewards. The 32% success rate is modest but demonstrates cross-domain transfer—discrete grids to 12-DoF continuous locomotion—with zero core code changes.

---

## 6. Analysis

### 6.1 Emergent Design Patterns

Across tasks and domains, evolved interfaces consistently exhibit:

**Phase conditioning.** On multi-phase tasks, the LLM introduces explicit phase flags and switches observation features based on task progress. This is notable because the LLM is never instructed to use phases—it discovers this structure through evolutionary pressure.

**Relative over absolute features.** Evolved observations prefer relative distances, normalized offsets, and direction-to-target vectors over raw positions. On the Go1 task, the observation computes direction-to-origin in the body frame rather than providing world-frame coordinates—a non-trivial coordinate transformation the LLM derives.

**Progress-based reward shaping.** Evolved rewards consistently compute `distance_prev - distance_next` with phase-appropriate weighting, providing dense signal for spatial progress.

### 6.2 Feature Quality over Quantity

The hard task illustrates that compact, task-relevant observations outperform large generic ones. The full-mode interface uses 147 dimensions and achieves 78.6%; the obs-only interface uses 472 dimensions (3.2× more) but achieves only 30.6% (2.6× less). When observations are co-evolved with rewards, the system learns what information *matters*; without reward co-evolution, it hedges by including everything.

### 6.3 Cost

| Task | LLM Cost | Wall Time |
|------|:--------:|:---------:|
| Easy | $3.88 | 52 min |
| Medium | $5.70 | 126 min |
| Hard | $11.22 | 99 min |
| Go1 | $3.05 | 5.4 hrs |

Total LLM cost for the complete grid-world evaluation (3 tasks × 4 modes × 10 seeds) is $21.55. Individual evolution runs cost $3–11 in LLM queries and 1–5 hours of single-GPU compute. These costs compare favorably to the days of expert iteration typically required for manual interface design.

---

## 7. Related Work

**LLM-guided reward design.** Eureka (Ma et al., 2024) evolves reward functions via iterative LLM refinement, achieving strong results across 29 IsaacGym tasks. Text2Reward (Xie et al., 2024) generates reward code from language descriptions. Both assume fixed observations. Our reward-only ablation is functionally equivalent to Eureka—we show it fails when observations are inadequate (2.2% on medium, 5.0% on hard), motivating joint optimization.

**Joint observation-reward evolution.** LERO (Wei et al., 2025) is the closest concurrent work, evolving joint observation-reward components for multi-agent RL. The key architectural difference: LERO *augments* existing observations with auxiliary features (preserving the base observation), while our approach *replaces* the entire observation from raw state. This distinction matters—LERO assumes a useful base observation exists; we make no such assumption, enabling discovery from scratch.

**LLM for RL components.** ONI (Zheng et al., 2025) generates intrinsic rewards for NetHack exploration. Action Mapping (Theile et al., 2025) transforms the action interface for constrained RL. These address single components; we address observation and reward jointly. Combined with action mapping, full interface evolution (observation + reward + action) is a natural future direction.

**Evolutionary program synthesis.** FunSearch (Romera-Paredes et al., 2024) and OpenELM (Lehman et al., 2024) demonstrate LLM-guided program evolution for mathematics and open-ended discovery. We adapt quality-diversity search (MAP-Elites; Mouret & Clune, 2015) to the MDP interface domain with domain-specific crash filtering, cascade evaluation, and training-feedback prompts.

**Automated RL.** AutoRL (Parker-Holder et al., 2022) surveys automated RL design. Meta-learning approaches (Zheng et al., 2020) optimize reward/observation parameters via gradients within fixed architectures. Our approach uses LLMs as the search operator, enabling open-ended code-level discovery—the evolved interfaces contain coordinate transformations, phase logic, and spatial reasoning that are outside the reach of parametric optimization.

---

## 8. Discussion and Limitations

**When does joint optimization help?** Our results suggest a clear pattern: joint optimization is most valuable when (a) the task has multiple phases requiring temporal credit assignment, and (b) the default observation lacks task-relevant structure. On single-phase tasks with adequate built-in rewards (our medium task), observation-only evolution suffices. This provides practitioners a diagnostic: if obs-only works, the bottleneck is the observation; if it doesn't, joint optimization is likely needed.

**Limitations.** (1) Our grid-world tasks, while multi-phase, are relatively simple compared to environments like Atari or real-world robotics. Scaling to higher-dimensional observations and longer horizons is untested. (2) Continuous control results are preliminary—Go1 achieves 32%, substantially below grid-world performance. More iterations, curriculum design, or stronger LLMs may be needed. (3) The system requires a human-written API reference (~2 pages) per environment. While far less effort than manual interface design, this is not fully zero-shot. (4) We evaluate with a single LLM (Claude Sonnet 4.6); performance may vary across models. (5) Some conditions show high variance (obs-only hard: $\pm$28.8%), and larger seed counts would tighten confidence intervals.

---

## 9. Conclusion

We studied the joint design of observation mappings and reward functions in RL—the MDP interface—and found that observation design is frequently the primary bottleneck, not reward design. On complex multi-phase tasks, joint observation-reward evolution achieves 78.6% success where the best single-component approach achieves 30.6%, because jointly evolved interfaces discover tightly coupled phase-conditioned observations and phase-gated rewards. The same system transfers from discrete grid worlds to continuous MuJoCo control at $3–11 per run. Our results suggest that the field's focus on reward design alone addresses only part of the interface design problem, and that joint optimization of what the agent sees and what it optimizes deserves greater attention.

---

## References

- Icarte, R. T., et al. (2022). Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning. *JAIR*.
- Lehman, J., et al. (2024). Evolution through Large Models. *Nature*.
- Ma, Y. J., et al. (2024). Eureka: Human-Level Reward Design via Coding Large Language Models. *ICLR*.
- Mouret, J.-B. & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.
- Muslimani, C., et al. (2025). Towards Improving Reward Design in RL: A Reward Alignment Metric. *RLJ*.
- Parker-Holder, J., et al. (2022). Automated Reinforcement Learning (AutoRL): A Survey and Open Problems. *JAIR*.
- Romera-Paredes, B., et al. (2024). Mathematical discoveries from program search with large language models. *Nature*.
- Theile, M., et al. (2025). Action Mapping for RL in Continuous Environments with Constraints. *RLJ*.
- Wei, Y., et al. (2025). LERO: LLM-driven Automatic Environment and Reward Design for RL in MARL. *arXiv:2503.21807*.
- Xie, T., et al. (2024). Text2Reward: Reward Shaping with Language Models for RL. *ICLR*.
- Zheng, Q., et al. (2025). Online Intrinsic Rewards for Decision Making Agents from LLM Feedback. *RLJ*.
- Zheng, Z., et al. (2020). What Can Learned Intrinsic Rewards Capture? *ICML*.

---

## Appendix A: Evolved Interface — Hard Task (Full Mode)

The best interface for the hard task (4-room pickup + place, 147 dimensions, 76% success). Key components of `get_observation`:

```python
def get_observation(state):
    # Phase encoding — tracks multi-step task progress
    holding_green_ball = ((pocket[0] == BALL) & (pocket[1] == GREEN)).astype(float)
    phase0 = (~holding & pyramid_exists).astype(float)  # seeking pyramid
    phase1 = holding_green_ball                           # carrying to hex
    phase2 = ball_adj_to_hex                              # placed successfully

    # Dynamic target — switches based on current phase
    target_y = jax.lax.select(holding > 0.5, hex_y, pyr_y)
    target_x = jax.lax.select(holding > 0.5, hex_x, pyr_x)

    # Best placement waypoint — closest floor tile adjacent to hex
    for d in range(4):
        neighbor = hex_pos + DIRECTIONS[d]
        is_valid = (grid[ny, nx, 0] == FLOOR)
        dist = manhattan(agent_pos, neighbor)
        best = select(is_valid & (dist < best_dist), neighbor, best)

    # 147 dims: agent(8) + objects(18) + front(8) + neighbors(24)
    #         + hex_neighbors(20) + placement(4) + phases(6)
    #         + target(4) + local_view(50) + putdown_valid(8)
```

Key components of `compute_reward`:

```python
def compute_reward(state, action, next_state):
    # Milestone bonuses at phase transitions
    pickup_reward  = select(just_picked_up, +5.0, 0.0)
    success_reward = select(placed_adjacent, +20.0, 0.0)
    wrong_penalty  = select(placed_wrong,    -5.0, 0.0)

    # Phase-gated progress shaping
    pyr_shaping   = select(~holding, (d_prev - d_next) * 3.0, 0.0)  # phase 0
    hex_shaping   = select(holding,  (d_prev - d_next) * 3.0, 0.0)  # phase 1
    place_shaping = select(holding,  (p_prev - p_next) * 1.5, 0.0)  # phase 1
```

## Appendix B: Evolved Interface — Go1 Push Recovery (Full Mode)

Best interface: 61 dimensions, 32% success.

**Observation** (selected features):
- Gravity vector and gyroscope in body frame (tanh-normalized by 5.0)
- XY position from origin (tanh-normalized by 0.5m)
- Direction to origin in body frame: `[cos(-h)*(-x) - sin(-h)*(-y), sin(-h)*(-x) + cos(-h)*(-y)]`
- Velocity projected toward origin: `-dot(vel_xy, pos_xy / norm(pos_xy))`
- Push force: normalized direction + `tanh(magnitude / 200)`
- 12 joint positions + 12 joint velocities (tanh-normalized) + 12 previous actions

**Reward** (selected terms):
- Upright: `clip(upvector_z, 0, 1)²` — quadratic for sharp upright/fallen distinction
- Position return: `exp(-2 * dist)` — gated on upright score
- Progress: `clip((dist_prev - dist_next) * 20, -2, 2)`
- Velocity penalty: `-0.2 * |vel| * exp(-5 * dist)` — active only near origin
- Fall penalty: `-3 * clip(0.3 - upvector_z, 0, 0.3) / 0.3`

## Appendix C: Hyperparameters

**Table C1: XLand-MiniGrid training.**

| Parameter | Easy | Medium | Hard |
|-----------|:----:|:------:|:----:|
| Timesteps (short / full) | 500K / 1M | 1M / 2M | 3M / 5M |
| Parallel envs | 8192 | 8192 | 8192 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Discount $\gamma$ | 0.99 | 0.99 | 0.99 |
| GAE $\lambda$ | 0.95 | 0.95 | 0.95 |
| Entropy coeff | 0.01 | 0.01 | 0.01 |
| Network | MLP+GRU (256h, 512r) | MLP+GRU (256h, 512r) | MLP+GRU (256h, 512r) |
| Eval episodes | 80 | 80 | 80 |
| Seeds per condition | 10 | 10 | 10 |

**Table C2: MuJoCo/Brax training.**

| Parameter | Go1 Push Recovery | Panda Tracking |
|-----------|:-----------------:|:--------------:|
| Timesteps (short / full) | 6M / 30M | 5M / 15M |
| Parallel envs | 4096 | 2048 |
| Learning rate | 0.0003 | 0.0005 |
| Discount $\gamma$ | 0.97 | 0.97 |
| Entropy coeff | 0.01 | 0.015 |
| Policy network | MLP [32,32,32,32] | MLP [32,32,32,32] |
| Value network | MLP [256]×5 | MLP [256]×5 |
| Episode length | 500 | 500 |
| Cascade threshold | 5% | 2% |
| Seeds (cascade) | 3 | 3 |
