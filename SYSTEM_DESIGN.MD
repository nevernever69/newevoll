# MDP Interface Discovery via LLM-Guided Evolutionary Search

## 1. Research Question

Can the MDP interface — the observation mapping, action mapping, and reward function — be treated as a learnable object? Can LLM-guided meta-optimization automatically construct effective task formulations that induce desired behavior in environments where no reward function is provided, given only a natural language task description and access to the environment's library/API?

### Sub-Questions

1. Does jointly discovering the full interface (obs + action + reward) outperform discovering reward alone?
2. What structural properties do discovered interfaces exhibit compared to human-designed ones?
3. Do discovered interfaces generalize to held-out environment configurations, noise, and layout changes?
4. Which interface components (observation, action, reward) contribute most to task performance?

---

## 2. System Overview

### What the System Does

Given:
- A **task description** in natural language (e.g., "navigate to a hidden goal in a grid with obstacles")
- A **library context document** (curated API reference for the environment framework — state variables, action spaces, wrapper conventions, code patterns)
- A **ground-truth evaluator** (a script that runs a trained policy and returns a task-completion metric the LLM never sees)

The system produces:
- A complete **MDP interface** — three functions that define how the RL agent perceives, acts in, and is rewarded by the environment — discovered entirely by the system with no human-designed starting point.

### What Makes This Different From Prior Work

- **EUREKA, Text2Reward, L2R**: These systems evolve or generate reward functions starting from existing code or structured reward templates. Our system generates the **full MDP interface from scratch** — not just reward, but also observation processing and action transformation.
- **No seed program**: The system does not receive a baseline MDP to mutate. It constructs the first candidate from nothing but the task description and library knowledge.
- **Joint optimization**: Observation, action, and reward are co-evolved, allowing the system to discover synergies between interface components that reward-only methods cannot find.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUTS                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Task          │  │ Library Context  │  │ Ground-Truth  │  │
│  │ Description   │  │ Document         │  │ Evaluator     │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬───────┘  │
└─────────┼──────────────────┼─────────────────────┼──────────┘
          │                  │                     │
          ▼                  ▼                     │
┌─────────────────────────────────────┐            │
│       LLM GENERATION ENGINE         │            │
│                                     │            │
│  Prompt = task_desc + lib_context   │            │
│        + population feedback        │            │
│        + error artifacts            │            │
│                                     │            │
│  Output = MDP Interface Code        │            │
│    - get_observation(state)         │            │
│    - transform_action(action)       │            │
│    - compute_reward(state, action,  │            │
│                     next_state)     │            │
└──────────────┬──────────────────────┘            │
               │                                   │
               ▼                                   │
┌──────────────────────────────────┐               │
│        CRASH FILTER              │               │
│                                  │               │
│  Stage 0: Syntax check           │               │
│  Stage 1: Import/dependency check│               │
│  Stage 2: Dry-run with dummy     │               │
│           state (no RL training) │               │
│                                  │               │
│  FAIL → error artifacts fed back │               │
│         to LLM for next attempt  │               │
└──────────────┬───────────────────┘               │
               │ (only code that runs)             │
               ▼                                   │
┌──────────────────────────────────┐               │
│     CASCADE RL TRAINING          │               │
│                                  │               │
│  Stage 1: Short training         │               │
│    (~500k env steps)             │               │
│    Threshold: must show learning │               │
│                                  │               │
│  Stage 2: Full training          │               │
│    (~5-10M env steps)            │               │
│    Only for candidates that      │               │
│    passed Stage 1                │               │
└──────────────┬───────────────────┘               │
               │ (trained policy)                  │
               ▼                                   ▼
┌──────────────────────────────────────────────────────┐
│           GROUND-TRUTH EVALUATION                     │
│                                                       │
│  Run trained policy in environment                    │
│  Measure: success rate, distance-to-goal,             │
│           path optimality, task completion             │
│                                                       │
│  This score is the evolutionary FITNESS.              │
│  The LLM and RL agent NEVER see this metric.          │
└──────────────┬────────────────────────────────────────┘
               │ (fitness score + evaluation artifacts)
               ▼
┌──────────────────────────────────────────────────┐
│        EVOLUTIONARY DATABASE (MAP-Elites)         │
│                                                   │
│  Population of MDP interfaces mapped to feature   │
│  grid (reward complexity, obs dimensionality,     │
│  action transform type, etc.)                     │
│                                                   │
│  Island-based evolution:                          │
│    - Multiple isolated populations                │
│    - Periodic migration between islands           │
│    - Maintains diversity                          │
│                                                   │
│  Stores per program:                              │
│    - MDP interface code                           │
│    - Fitness score                                │
│    - Training artifacts (learning curves, errors) │
│    - Feature coordinates                          │
└──────────────┬────────────────────────────────────┘
               │
               │ (top programs + diverse programs
               │  + failure artifacts)
               │
               └──────────► back to LLM GENERATION ENGINE
                            (next evolutionary iteration)
```

### 3.2 MDP Interface Template

The LLM generates a single Python file containing three functions with fixed signatures. This is the exact interface the system expects:

```python
"""
MDP Interface for: {task_description}
Generated by evolutionary iteration {iteration_number}
"""

import jax
import jax.numpy as jnp
from typing import Any

# Type alias for environment state (provided by library)
State = Any


def get_observation(state: State) -> jnp.ndarray:
    """
    Extract and transform observations from the raw environment state.

    The RL agent sees ONLY what this function returns.
    You decide what information to include, exclude, transform, or combine.

    Args:
        state: Raw environment state object (see library context for available fields)

    Returns:
        observation: 1D JAX array of floats that the policy network receives as input
    """
    # LLM generates this implementation
    pass


def transform_action(action: jnp.ndarray) -> jnp.ndarray:
    """
    Transform the policy's output action before applying it to the environment.

    The policy outputs a raw action vector. This function can scale, clip,
    remap, or otherwise transform it before it reaches the environment.

    Args:
        action: Raw action vector from the policy network

    Returns:
        transformed_action: Action vector to apply to the environment
    """
    # LLM generates this implementation
    pass


def compute_reward(
    state: State,
    action: jnp.ndarray,
    next_state: State
) -> float:
    """
    Compute the reward signal for the RL agent.

    This is the ONLY reward the agent sees during training.
    Design this to induce the desired behavior described in the task.

    Args:
        state: Environment state before the action
        action: Action that was taken
        next_state: Environment state after the action

    Returns:
        reward: Scalar reward value
    """
    # LLM generates this implementation
    pass
```

**Design decisions:**
- Fixed function signatures eliminate structural errors and make outputs pluggable.
- The LLM has full freedom WITHIN each function — arbitrary Python/JAX logic, helper variables, conditionals, mathematical expressions.
- The LLM can add helper functions above the three main functions if needed.
- All functions must be JAX-compatible (no Python control flow on traced values, use jnp not np, etc.) — this constraint is stated in the library context document.

### 3.3 Library Context Document

One curated document per environment framework. Written manually. Contains:

**Section 1: Framework Overview**
- What the framework is (Brax, custom GridWorld engine)
- How environments work at a high level
- JAX constraints and conventions

**Section 2: State Object Reference**
- Every field accessible on the state object
- Data types and shapes
- Value ranges where known
- Example: `state.pipeline_state.q` — joint positions, shape (n_joints,), range [-pi, pi]

**Section 3: Action Space**
- Action vector dimensionality
- What each action dimension controls
- Valid ranges
- How actions are applied to the environment

**Section 4: Wrapper Pattern**
- How the MDP interface plugs into the training loop
- Import conventions
- How get_observation output connects to the policy network input
- How compute_reward output is used by PPO

**Section 5: Code Conventions**
- JAX-compatible patterns (jnp.where instead of if/else, etc.)
- Common pitfalls
- Valid imports

**Section 6: Available Utilities**
- Math functions available
- State access helpers
- Normalization utilities

The document does NOT contain:
- Any reward function examples
- Any description of what "good" behavior looks like
- The ground-truth evaluation metric
- Hints about what observations matter for any specific task

### 3.4 LLM Generation Engine

#### Prompt Structure — Iteration 1 (From Scratch)

```
SYSTEM: You are an expert RL engineer. You design MDP interfaces —
observation functions, action transforms, and reward functions —
for reinforcement learning agents.

CONTEXT:
{library_context_document}

TASK:
{task_description}

INSTRUCTIONS:
Generate a complete MDP interface (get_observation, transform_action,
compute_reward) that will allow a PPO agent to learn the described task.

You must output a single Python file following this exact template:
{mDP_interface_template}

Think carefully about:
- What observations the agent needs to solve this task
- Whether the action space needs transformation
- What reward signal will guide the agent toward the desired behavior

Output ONLY the Python file. No explanation.
```

#### Prompt Structure — Iteration N (Evolutionary)

```
SYSTEM: You are an expert RL engineer designing MDP interfaces.

CONTEXT:
{library_context_document}

TASK:
{task_description}

PREVIOUS BEST INTERFACES:
--- Interface #{id} (fitness: {score}) ---
{code}
--- END ---

--- Interface #{id} (fitness: {score}) ---
{code}
--- END ---

DIVERSE INTERFACES (different approaches):
--- Interface #{id} (fitness: {score}) ---
{code}
--- END ---

RECENT FAILURES:
--- Interface #{id} (error) ---
{code}
Error: {stack_trace_or_error_message}
--- END ---

ARTIFACTS FROM BEST PROGRAMS:
- Training curve showed reward plateauing at step 200k
- Agent learned to move but never reached the goal
- {other_feedback_from_evaluation}

INSTRUCTIONS:
Based on the above attempts and their results, generate an improved
MDP interface. Consider:
- What worked in high-scoring interfaces
- What failed and why
- Novel approaches not yet tried

Output ONLY the Python file. No explanation.
```

#### Prompt Sampling Strategy (from OpenEvolve)

For each generation:
- **Top K programs** (K=2-3): Highest fitness scores in the population. Always includes the absolute best.
- **Diverse D programs** (D=1-2): Selected from different MAP-Elites cells to encourage exploration of different approaches.
- **Recent failures** (F=1-2): Programs that crashed or scored poorly, with their error messages. Teaches the LLM what NOT to do.
- **Artifacts**: Qualitative feedback from evaluation (training curves, behavioral observations).

### 3.5 Crash Filter

Before any RL training (which is expensive even with JAX), candidate MDP interfaces pass through a multi-stage filter:

**Stage 0 — Syntax Check**
- Parse the generated Python file
- Check for syntax errors
- Verify all three functions are defined with correct signatures
- Cost: ~0ms

**Stage 1 — Import & Dependency Check**
- Attempt to import the module
- Verify all imports resolve
- Check JAX compatibility (no numpy where jnp is needed)
- Cost: ~1s

**Stage 2 — Dry Run**
- Create a dummy environment state
- Call get_observation(dummy_state) — verify it returns a valid array
- Call transform_action(dummy_action) — verify shape compatibility
- Call compute_reward(dummy_state, dummy_action, dummy_state) — verify it returns a scalar
- Cost: ~2-5s

**On failure at any stage:**
- Record the error message and stack trace
- Store as artifact in the database
- Feed back to LLM in next generation prompt
- Do NOT proceed to RL training

**Retry logic:**
- On crash, optionally give the LLM one immediate retry with the error message before moving to the next evolutionary candidate
- This helps with simple fixes (wrong attribute name, shape mismatch) without wasting an evolutionary iteration

### 3.6 Cascade RL Training

Candidates that pass the crash filter enter RL training in stages:

**Stage 1 — Short Training Run**
- PPO training for ~500k environment steps
- Single random seed
- Timeout: environment-dependent (seconds for JAX-compiled Brax)
- Threshold: must show non-zero learning signal (reward increasing over time, or ground-truth metric above random baseline)
- Purpose: filter out reward functions that produce no learning signal

**Stage 2 — Full Training Run**
- PPO training for 5-10M environment steps (environment-dependent)
- 3 random seeds for statistical reliability
- Ground-truth evaluator runs on the final trained policy (averaged across seeds)
- Full training artifacts collected (learning curves, final behavior)

**Score computation:**
- Fitness = mean ground-truth metric across seeds
- Variance across seeds recorded as metadata (high variance = unreliable interface)

### 3.7 Evolutionary Database

Adapted from OpenEvolve's MAP-Elites + island-based evolution.

**Feature Dimensions for MAP-Elites Grid:**
- **Reward complexity**: Number of terms/components in compute_reward (simple scalar vs multi-component shaped reward)
- **Observation dimensionality**: Size of the observation vector returned by get_observation (minimal vs rich)
- **Action transform type**: Identity / linear scaling / nonlinear remapping
- **Fitness score**: The ground-truth evaluation metric

This creates a 2D or 3D grid where each cell stores the best MDP interface with those characteristics. This maintains diversity — the system keeps both simple and complex approaches alive.

**Island-Based Evolution:**
- 3-5 islands, each with its own MAP-Elites grid
- Islands evolve independently
- Every N generations, top programs migrate between islands (ring topology)
- Prevents premature convergence on one approach

**What's stored per program:**
- Full MDP interface code
- Fitness score (ground-truth metric)
- Feature coordinates (where it sits in MAP-Elites grid)
- Training artifacts (learning curves, behavioral notes)
- Error artifacts (if crashed or performed poorly)
- Parent ID (which program inspired this one)
- Iteration number
- Island ID

### 3.8 Ground-Truth Evaluator

A separate script per environment that the LLM NEVER sees. Takes a trained policy and evaluates it on the actual task objective.

**GridWorld Example:**
```python
def evaluate(policy, env, n_episodes=100):
    successes = 0
    total_path_length = 0
    optimal_path_length = compute_optimal(env)  # BFS/A*

    for episode in range(n_episodes):
        state = env.reset()
        path_length = 0
        for step in range(max_steps):
            action = policy(state)
            state, _, done, _ = env.step(action)
            path_length += 1
            if done and state.at_goal:
                successes += 1
                total_path_length += path_length
                break

    return {
        "success_rate": successes / n_episodes,
        "path_optimality_gap": (total_path_length / successes) / optimal_path_length - 1.0,
    }
```

**Brax Example:**
```python
def evaluate(policy, env, n_episodes=100):
    total_distance = 0
    successes = 0

    for episode in range(n_episodes):
        state = env.reset()
        for step in range(max_steps):
            action = policy(get_default_obs(state))  # uses RAW state, not discovered obs
            state = env.step(state, action)

        final_dist = jnp.linalg.norm(state.pos - target_pos)
        total_distance += final_dist
        if final_dist < success_threshold:
            successes += 1

    return {
        "success_rate": successes / n_episodes,
        "mean_distance_to_goal": total_distance / n_episodes,
    }
```

**Important:** The ground-truth evaluator uses the trained policy but evaluates on the RAW task metric. The evolved reward function shaped training, but the evaluation measures actual task completion.

---

## 4. Environments

### 4.1 Phase 1: Custom JAX-Based GridWorld

All GridWorld environments are implemented in JAX for fast training in the inner loop.

#### Environment 1: Hidden Goal Navigation

**Description:** Agent must navigate a grid to find a hidden goal. Distractor objects are placed throughout the grid. The agent has partial observability (can only see a local window around itself).

**Properties:**
- Grid size: 10x10 (in-distribution), 15x15 and 20x20 (generalization tests)
- Partial observability: 5x5 local view
- Distractor objects: 3-5 per grid (look different from goal but are visible)
- Hidden goal: Not visible until agent is adjacent or on top of it
- Stochastic initial layouts: Agent start position, goal position, distractor positions, wall positions randomized each episode
- Episode length: 200 steps

**State available to MDP interface:**
- agent_position: (x, y)
- local_view: 5x5 grid of cell types (wall, empty, distractor, goal-if-adjacent)
- step_count: current timestep
- visited_cells: boolean grid of where agent has been
- distractor_positions: list of (x, y)

**Action space:**
- 4 discrete actions: up, down, left, right

**Ground-truth metric:**
- Success rate (goal reached within episode length)
- Path optimality gap (actual path length / optimal path length - 1)

**Why this is hard without a good MDP interface:**
- Sparse reward (goal reached = +1) provides almost no learning signal in a 10x10 grid with partial observability
- The agent needs to learn exploration behavior, which requires reward shaping for visiting new cells, moving toward unexplored areas, etc.
- Good observation processing might include: encoding visited cells as features, distance to nearest unexplored area, relative distractor positions
- The system must discover these structures from scratch

#### Environment 2: Multi-Key Door Navigation

**Description:** Agent must collect keys in a specific order to unlock doors and reach the final goal. Requires sequential reasoning.

**Properties:**
- Grid size: 12x12
- 2-3 keys of different colors, corresponding locked doors
- Keys must be collected in order (key A opens door A, which leads to key B, etc.)
- Partial observability: 5x5 local view
- Stochastic key/door placement

**State available to MDP interface:**
- agent_position: (x, y)
- local_view: 5x5 grid
- keys_collected: boolean array [has_key_A, has_key_B, ...]
- doors_unlocked: boolean array
- step_count: current timestep

**Action space:**
- 4 discrete actions: up, down, left, right
- Pickup/unlock is automatic when agent is on key/door

**Ground-truth metric:**
- Task completion rate (reached final goal)
- Partial completion (fraction of keys collected)
- Step efficiency

**Why this is hard:**
- Requires hierarchical reward shaping (reward for each key, each door, final goal)
- Observation function must encode key/door state meaningfully
- Wrong reward structure (e.g., rewarding proximity to final goal before having keys) actively hurts learning

#### Environment 3: Predator Evasion with Resource Collection

**Description:** Agent must collect resources scattered across the grid while avoiding a predator that follows a deterministic pursuit policy. Episode ends if caught.

**Properties:**
- Grid size: 10x10
- 5-8 resources randomly placed
- 1 predator with deterministic chase behavior (moves toward agent each step with 80% probability)
- Full observability (agent sees entire grid)
- Episode length: 150 steps or until caught

**State available to MDP interface:**
- agent_position: (x, y)
- predator_position: (x, y)
- resource_positions: list of (x, y)
- resources_collected: int
- full_grid: 10x10 cell types
- step_count: current timestep

**Action space:**
- 4 discrete actions: up, down, left, right

**Ground-truth metric:**
- Resources collected per episode (mean)
- Survival rate (not caught)
- Combined: resources_collected * survival_bonus

**Why this is hard:**
- Conflicting objectives: collect resources (approach them) vs. avoid predator (flee)
- Reward must balance risk/reward tradeoff
- Observation function should encode relative threat level, nearest safe resource, escape routes
- Action remapping could encode "move toward nearest safe resource" as a macro

### 4.2 Phase 2: Custom Brax Continuous Control

All environments built using Brax with JAX compilation.

#### Environment 4: Target Reaching with Obstacles

**Description:** A simple articulated robot arm (or point-mass with dynamics) must reach a target position while avoiding obstacles. No predefined reward.

**Properties:**
- 2D or 3D workspace
- Robot: articulated arm with 2-3 joints OR point-mass with momentum
- Target: fixed or randomized position
- Obstacles: 2-3 static obstacles that the agent must navigate around
- Continuous state and action spaces
- Randomized target and obstacle positions per episode

**State available to MDP interface:**
- joint_positions: array of joint angles
- joint_velocities: array of joint angular velocities
- end_effector_position: (x, y, z) position of the tip
- target_position: (x, y, z)
- obstacle_positions: list of (x, y, z, radius)
- contact_forces: array of contact sensor readings
- timestep: current step

**Action space:**
- Joint torques: continuous array, one per joint
- Range: [-1, 1] normalized

**Ground-truth metric:**
- Distance from end-effector to target at episode end
- Success rate (within threshold distance)
- Collision count (number of obstacle contacts)

**Why this is hard:**
- Continuous control with dynamics — reward shaping must account for velocity, not just position
- Obstacle avoidance requires negative reward components that must be balanced with reaching reward
- Observation function must decide: raw joint angles? relative target vector? obstacle proximity features?
- Action transform might need to scale torques or add damping

#### Environment 5: Object Pushing to Target Region

**Description:** A robot must push an object (box/sphere) to a target region on the floor. Requires contact-rich manipulation.

**Properties:**
- Robot: simple 2-joint pusher or point-mass with collision body
- Object: box or sphere with physics
- Target region: marked area on the ground
- Randomized initial object and target positions

**State available to MDP interface:**
- robot_position, robot_velocity
- object_position, object_velocity
- target_region_center, target_region_radius
- contact_between_robot_and_object: boolean + force
- timestep

**Action space:**
- Robot movement forces: continuous 2D or 3D
- Range: [-1, 1] normalized

**Ground-truth metric:**
- Distance from object to target center at episode end
- Success rate (object within target region)
- Efficiency (steps to success)

**Why this is hard:**
- Indirect control — agent must manipulate object through contact
- Reward must incentivize approach-object, contact-object, push-toward-target as sequential subgoals
- Observation function must encode robot-object-target geometric relationships
- Naive reward (distance from robot to target) fails because agent goes to target without the object

#### Environment 6: Balanced Locomotion

**Description:** A simple multi-legged creature must learn to walk forward while maintaining balance. No predefined gait or reward.

**Properties:**
- Robot: 4-legged simple creature (ant-like but custom, not standard Ant)
- Task: move forward (positive x direction) without falling
- Terrain: flat
- Randomized initial joint positions (small perturbations)

**State available to MDP interface:**
- torso_position: (x, y, z)
- torso_orientation: quaternion
- torso_velocity: (vx, vy, vz)
- torso_angular_velocity: (wx, wy, wz)
- joint_positions: array per joint
- joint_velocities: array per joint
- foot_contacts: boolean array per foot
- timestep

**Action space:**
- Joint torques: continuous array
- Range: [-1, 1] normalized

**Ground-truth metric:**
- Forward distance traveled (x displacement)
- Survival time (didn't fall over)
- Combined: distance * survival_bonus

**Why this is hard:**
- High-dimensional continuous control
- Reward must balance forward progress, balance, energy efficiency, gait regularity
- Many possible observation encodings (relative joint angles, velocity features, contact patterns)
- Action transform could impose symmetry (left-right leg coupling) or scale torques
- Naive reward (just x-velocity) produces unstable, energy-wasteful gaits

---

## 5. Baselines

### 5.1 Sparse Reward Baseline

For each environment, train PPO with the most minimal possible reward:
- GridWorld: +1 on goal reached, 0 otherwise
- Brax reaching: -distance_to_target at final step only
- Brax pushing: -distance_object_to_target at final step only
- Brax locomotion: x_velocity (raw, unscaled)

Default observations: full raw state vector, no processing.
Default actions: raw policy output, no transformation.

This represents the "no reward engineering" baseline.

### 5.2 Human-Designed Baseline

For each environment, a human (you) designs a reasonable MDP interface:
- Shaped reward with 3-5 components (progress, penalties, bonuses)
- Sensible observation selection (relevant features, normalized)
- Appropriate action scaling

This should be what a competent RL engineer would write in 30-60 minutes. Document the time spent. This represents "standard practice."

### 5.3 Reward-Only Evolution Baseline

Use the full evolutionary system but only evolve compute_reward. Observations and action transforms are fixed to the same defaults as the sparse baseline.

This isolates the contribution of reward discovery and directly compares with EUREKA-style approaches (which only evolve reward).

### 5.4 Random Search Baseline

Generate N random MDP interfaces (same LLM, no evolutionary selection — just independent samples) and report the best one found. Same total LLM budget as the evolutionary system.

This tests whether evolution actually helps or if the LLM is good enough that random sampling works.

---

## 6. Ablation Studies

### 6.1 Interface Component Ablations

Run in all environments:

| Ablation | Observation | Action | Reward | Purpose |
|---|---|---|---|---|
| Full system | Evolved | Evolved | Evolved | Complete system |
| Reward-only | Default | Default | Evolved | Is reward alone sufficient? |
| Obs-only | Evolved | Default | Default (sparse) | Does obs processing help with sparse reward? |
| Action-only | Default | Evolved | Default (sparse) | Does action remapping help? |
| Obs + Reward | Evolved | Default | Evolved | Is action remapping needed? |
| Reward + Action | Default | Evolved | Evolved | Is obs processing needed? |

**Key hypothesis:** Full system > Reward-only > any single component. If Obs+Reward ≈ Full, action remapping doesn't matter (and that's a valid finding).

### 6.2 Generalization Ablations

Test discovered MDP interfaces on configurations NOT seen during evolution:

**GridWorld:**
- Larger grid sizes (15x15, 20x20 — trained on 10x10)
- More distractor objects (8-10 — trained with 3-5)
- Different wall layouts (held-out random seed)
- Observation corruption (random noise added to local view)
- Action stochasticity (10% random action instead of chosen)

**Brax:**
- Different target positions (held-out distribution)
- Added sensor noise (Gaussian noise on joint readings)
- Action noise (torque perturbations)
- Minor dynamics variation (±10% gravity, ±10% friction)
- Different initial conditions

**Protocol:**
1. Evolve MDP interface on the standard configuration
2. Freeze the discovered interface (no further evolution)
3. Retrain a fresh policy using the frozen interface on each generalization condition
4. Evaluate with ground-truth metric
5. Compare degradation across baselines

### 6.3 Evolutionary Method Ablations

| Ablation | Description | Purpose |
|---|---|---|
| No evolution (random) | Random LLM samples, no selection | Does evolution help? |
| No islands | Single population, no islands | Do islands help diversity? |
| No MAP-Elites | Standard best-only selection | Does quality-diversity help? |
| No error feedback | Don't show failures to LLM | Does error feedback help? |
| No cascade | Full training for all candidates | Does cascade save compute without hurting quality? |
| Smaller LLM | Use a weaker/smaller model | How much does LLM capability matter? |
| No artifact feedback | Don't show training curves/behavior | Does qualitative feedback help? |

### 6.4 Cold Start Analysis

Track across all environments:
- Iteration at which first runnable MDP interface is generated
- Iteration at which first positive-fitness MDP is found
- Iteration at which performance exceeds sparse reward baseline
- Iteration at which performance exceeds human-designed baseline (if ever)

Plot learning curves: fitness vs. evolutionary iteration. This shows how quickly the system bootstraps from nothing.

---

## 7. Evaluation Protocol

### 7.1 Metrics Reported

For each environment × method combination:

**Primary Metrics:**
- Final ground-truth score (mean ± std across 5 outer seeds)
- Learning speed: evolutionary iterations to reach 50% of human baseline performance
- Success rate (task completion)

**Secondary Metrics:**
- Path optimality gap (GridWorld)
- Energy efficiency (Brax locomotion)
- Collision count (Brax obstacle environments)
- Sample efficiency: total environment steps to reach threshold performance

**Per outer seed:** Run the full evolutionary process independently. Report mean ± std across outer seeds to capture variance from LLM stochasticity and evolutionary randomness.

**Per inner evaluation:** 3 RL training seeds per candidate evaluation in the full training stage. Report mean across inner seeds as the fitness score.

### 7.2 Statistical Rigor

- 5 independent evolutionary runs (outer seeds) per method per environment
- 3 RL training seeds per candidate evaluation
- Report mean ± standard deviation
- Paired t-test or Wilcoxon signed-rank test for significance between methods
- Effect size (Cohen's d) for key comparisons

### 7.3 Qualitative Analysis

For the best-discovered MDP interfaces:

**Reward Structure Visualization:**
- Plot reward components and their relative magnitudes
- Compare structure to human-designed baseline
- Identify novel reward terms the system discovered

**Observation Analysis:**
- What features does the discovered observation function include/exclude?
- Are there computed features (e.g., relative positions, velocities) not in the raw state?
- Does observation processing correlate with task requirements?

**Action Transform Analysis:**
- Does the system learn meaningful action scaling?
- Are there symmetry patterns (e.g., left-right leg coupling)?
- Or does it converge to identity (i.e., action remapping doesn't matter)?

**Evolution Trajectory:**
- Tree visualization of parent-child relationships
- How does the population diversify over time?
- When do qualitative shifts in strategy occur?

---

## 8. Implementation Plan

### 8.1 New Repository Structure

```
mdp-discovery/
├── mdp_discovery/                    # Core system
│   ├── __init__.py
│   ├── controller.py                 # Main evolution loop (adapted from OpenEvolve)
│   ├── database.py                   # MAP-Elites + islands (from OpenEvolve)
│   ├── evaluator.py                  # Crash filter + cascade RL training
│   ├── generator.py                  # LLM generation engine (NEW)
│   ├── config.py                     # Configuration (adapted from OpenEvolve)
│   ├── interface_template.py         # MDP interface template definition
│   ├── training/
│   │   ├── ppo.py                    # JAX PPO implementation
│   │   └── trainer.py                # Training loop wrapper
│   ├── llm/
│   │   ├── provider.py              # LLM API client (from OpenEvolve)
│   │   └── prompts.py               # Prompt templates (NEW)
│   └── utils/
│       ├── crash_filter.py           # Syntax/import/dry-run checks
│       └── artifacts.py              # Artifact collection
│
├── environments/                     # Custom environments
│   ├── gridworld/
│   │   ├── engine.py                 # JAX GridWorld engine
│   │   ├── hidden_goal.py            # Environment 1
│   │   ├── multi_key_door.py         # Environment 2
│   │   ├── predator_evasion.py       # Environment 3
│   │   └── ground_truth.py           # Ground-truth evaluators
│   └── brax/
│       ├── target_reaching.py        # Environment 4
│       ├── object_pushing.py         # Environment 5
│       ├── balanced_locomotion.py    # Environment 6
│       └── ground_truth.py           # Ground-truth evaluators
│
├── contexts/                         # Library context documents
│   ├── gridworld_context.md          # GridWorld API reference
│   └── brax_context.md              # Brax API reference
│
├── configs/                          # Experiment configurations
│   ├── default.yaml
│   ├── gridworld_hidden_goal.yaml
│   ├── gridworld_multi_key.yaml
│   ├── gridworld_predator.yaml
│   ├── brax_reaching.yaml
│   ├── brax_pushing.yaml
│   └── brax_locomotion.yaml
│
├── baselines/                        # Baseline MDP interfaces
│   ├── sparse/                       # Sparse reward baselines
│   └── human/                        # Human-designed baselines
│
├── experiments/                      # Experiment scripts
│   ├── run_evolution.py              # Main evolution runner
│   ├── run_baseline.py               # Run baselines
│   ├── run_ablation.py               # Run ablation studies
│   ├── run_generalization.py         # Run generalization tests
│   └── analyze_results.py            # Generate tables and figures
│
├── tests/                            # Test suite
│   ├── test_crash_filter.py
│   ├── test_evaluator.py
│   ├── test_generator.py
│   ├── test_environments.py
│   └── test_pipeline.py
│
├── results/                          # Output directory
│   ├── checkpoints/
│   ├── figures/
│   └── tables/
│
├── SYSTEM_DESIGN.md                  # This document
├── README.md
├── requirements.txt
└── setup.py
```

### 8.2 Build Order

**Step 1: Foundation**
- Set up new repository
- Port MAP-Elites database from OpenEvolve
- Port cascade evaluation framework from OpenEvolve
- Implement MDP interface template
- Implement crash filter

**Step 2: Environments**
- Build JAX GridWorld engine
- Implement Environment 1 (Hidden Goal)
- Implement ground-truth evaluator for Environment 1
- Write GridWorld library context document
- Verify: can a hand-written MDP interface train a working policy?

**Step 3: Generation Engine**
- Implement from-scratch generation prompts
- Implement evolutionary generation prompts
- Implement LLM API client
- Test: does the LLM produce runnable code from the context document?

**Step 4: Core Loop**
- Wire everything together: generation → crash filter → training → evaluation → database → next generation
- Run first end-to-end evolution on Environment 1
- Debug and tune cascade thresholds
- Validate cold start behavior

**Step 5: Remaining Environments**
- Build remaining GridWorld environments (2, 3)
- Build Brax environments (4, 5, 6)
- Write Brax library context document
- Verify each environment works end-to-end

**Step 6: Baselines**
- Implement sparse reward baselines for all environments
- Design and implement human baselines for all environments
- Implement reward-only evolution baseline
- Implement random search baseline

**Step 7: Experiments**
- Run main experiments (all environments × all methods × 5 seeds)
- Run ablation studies
- Run generalization tests
- Collect results

**Step 8: Analysis**
- Generate figures and tables
- Qualitative analysis of discovered interfaces
- Statistical significance tests
- Write paper

---

## 9. Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Library context delivery | Manual document | Reproducible, controlled, no exploration variance. Agentic exploration is future work. |
| MDP interface format | Three fixed-signature functions | Constrains structure enough that code runs, flexible enough for real discovery. |
| Inner-loop algorithm | PPO | Standard, well-understood, fast in JAX. Not the variable under study. |
| Evolutionary method | MAP-Elites + islands | Maintains diversity of approaches. From OpenEvolve, proven to work. |
| Environments | All custom, JAX-based | Avoids LLM training data bias. JAX for fast inner loop. |
| Ground-truth metric | Task-completion based | Clear, unambiguous, not gameable by reward hacking. |
| Evaluation seeds | 3 inner, 5 outer | Balances statistical reliability with compute cost. |
| Cascade training | 500k short / 5-10M full | Filters bad candidates cheaply. Thresholds validated empirically. |
| Crash filter | 3-stage pre-training check | Most LLM outputs won't run. Fail fast and cheap. |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Cold start failure (LLM can't generate runnable code) | System never bootstraps | Retry with error feedback. Simplify template if needed. Test early. |
| RL training noise masks evolutionary signal | Evolution selects randomly | Multiple inner seeds. Longer short-training stage. |
| Action remapping shows no benefit | Lose 1/3 of contribution | This is a valid negative result. Paper frames it as "reward + obs sufficient." |
| Custom environments too easy | Sparse baseline already works | Test difficulty during environment design. Add complexity until sparse fails. |
| Custom environments too hard | Nothing works including human baseline | Verify human baseline achieves reasonable performance before running evolution. |
| LLM generates memorized reward functions | Undermines "from scratch" claim | Custom environments not in training data. Analyze generated code for novelty. |
| Evolutionary search doesn't beat random LLM sampling | Evolution component unnecessary | Report honestly. Contribution becomes "LLM-based MDP generation" without evolution. |
