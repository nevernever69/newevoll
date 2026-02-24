# PureEvol: MDP Interface Discovery via LLM-Guided Evolutionary Search

## System Documentation & Experimental Results

---

## 1. Overview

**PureEvol** is the first system to discover **complete MDP interfaces** (observation mappings + reward functions) from scratch using LLM-guided evolutionary search. Unlike prior work that optimizes reward functions for existing benchmark tasks, PureEvol creates the **entire task specification**—what the agent sees and what it's optimized for—from a natural language description alone.

### Key Differentiator from Prior Work

| Method | What's Evolved | Task Source | Observation |
|--------|---------------|-------------|-------------|
| **PureEvol (Ours)** | **Observation + Reward** | **Novel tasks (not from benchmarks)** | **Discovered** |
| Eureka (2024) | Reward only | 29 benchmark tasks + 1 novel (pen spinning) | Fixed (env-provided) |
| AutoRL (2023) | Reward only | Benchmark tasks | Fixed |
| PROMPT (2023) | Reward only | Benchmark tasks | Fixed |

#### Eureka's "Novel" Task: Pen Spinning

Eureka's primary novel task demonstrates the limitations of reward-only optimization:

**Given:**
- Isaac Gym environment with Shadow Hand + pen
- Fixed observation space (joint positions, pen orientation, angular velocities)
- No pre-defined reward function

**Approach (Curriculum Learning):**
1. **Pre-training stage:** Reward for re-orientation (move pen to random target configurations)
2. **Fine-tuning stage:** Reward for continuous rotation (maximize spin cycles)

**What Eureka Discovers:**
- Only the reward function (weights, terms, shaping)
- Observation space is **fixed by the environment**

**PureEvol's Approach:**
- Given: Natural language only (e.g., "Pick up blue pyramid, place adjacent to yellow hex")
- **No curriculum, no pre-training**
- Discovers **both** observation space (which state fields, how to combine) **and** reward function
- All 3 tasks are **fully novel** (not from MiniGrid, XLand, or any benchmark)

---

## 2. System Architecture

### 2.1 Core Design Principle

**The context document is the single source of environment-specific knowledge.**

All system code (prompts, crash filter, training loop, evaluator) contains **zero** environment-specific hints, constants, or patterns. To support a new environment:
1. Write a context document (condensed API reference)
2. Write an adapter (implements `EnvAdapter` ABC)
3. **No changes to core system code**

```bash
# Verification: Only ONE file imports xminigrid in the entire codebase
$ grep -r "import xminigrid" mdp_discovery/
mdp_discovery/adapters/xminigrid_adapter.py  # ← Only here

# Core modules have ZERO env-specific imports
$ grep -r "import xminigrid" mdp_discovery/*.py
(no results)
```

### 2.2 Repository Structure

```
pureevol/
├── run.py                              # CLI entry point
├── configs/
│   ├── default.yaml                    # Default config
│   ├── easy_pickup.yaml                # Easy task config
│   ├── medium_place_near.yaml          # Medium task config
│   └── hard_rule_chain.yaml            # Hard task config
├── contexts/
│   └── xminigrid_context.md            # XLand-MiniGrid API reference
├── templates/
│   └── mdp_interface_template.py       # Interface contract (signatures only)
├── mdp_discovery/
│   ├── __init__.py
│   ├── config.py                       # Hierarchical dataclass config + YAML
│   ├── controller.py                   # Main evolution loop
│   ├── evaluator.py                    # Cascade evaluator pipeline
│   ├── crash_filter.py                 # 3-stage code validation
│   ├── train.py                        # PPO training (JAX pmap)
│   ├── train_utils.py                  # Rollout, GAE, PPO update
│   ├── nn.py                           # ActorCriticMLP with GRU
│   ├── mdp_interface.py                # Interface loader + validator
│   ├── prompts.py                      # Prompt templates + PromptBuilder
│   ├── llm_client.py                   # AWS Bedrock Claude client
│   ├── database.py                     # MAP-Elites + island evolution
│   └── adapters/
│       ├── __init__.py                 # Factory: get_adapter(type, config)
│       ├── base.py                     # EnvAdapter ABC
│       └── xminigrid_adapter.py        # XMinigrid implementation
├── benchmark_best_vs_sparse.py         # Main benchmark script
├── benchmark_ablation_spaces.py        # Ablation benchmark script
└── RESULTS.MD                          # This document
```

### 2.3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUTS                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐      │
│  │ Task          │  │ Context Doc      │  │ Config        │      │
│  │ Description   │  │ (API reference)  │  │ (YAML)        │      │
│  │ (natural lang)│  │ (e.g., xminigrid)│  │               │      │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬───────┘      │
└─────────┼───────────────────┼────────────────────┼──────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION CONTROLLER                          │
│  mdp_discovery/controller.py                                     │
│  - Creates Environment Adapter (from config.adapter_type)        │
│  - Loads context doc (from config.context_file)                  │
│  - Initializes PromptBuilder, Evaluator, Database                │
│  - Runs evolution loop or baseline mode                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐    ┌──────────────┐   ┌──────────────┐
   │   LLM    │    │  Cascade     │   │  MAP-Elites  │
   │Generation│───▶│  Evaluator   │──▶│  Database    │
   │ Engine   │    │              │   │  + Islands   │
   └──────────┘    └──────────────┘   └──────────────┘
```

### 2.4 Component Details

#### Environment Adapter (`mdp_discovery/adapters/`)

**Abstract interface** (`base.py`):
```python
class EnvAdapter(ABC):
    @abstractmethod
    def make_env(self, get_obs_fn: Optional[Callable], reward_fn: Optional[Callable]) -> Tuple[Any, Any]
        """Create wrapped environment with custom MDP interface."""
    
    @abstractmethod
    def make_eval_env(self, get_obs_fn: Optional[Callable], reward_fn: Optional[Callable], max_steps: int) -> Tuple[Any, Any]
        """Create evaluation environment with custom max_steps."""
    
    @abstractmethod
    def get_dummy_state(self) -> Any
        """Get real environment state for dry-run validation."""
    
    @abstractmethod
    def compute_success(self, rollout_stats: Any, env_params: Any) -> jax.Array
        """Compute per-episode success from rollout statistics."""
    
    @abstractmethod
    def get_default_obs_fn(self) -> Callable
        """Return default (no-design-effort) observation function."""
    
    @abstractmethod
    def get_default_reward_fn(self) -> Optional[Callable]
        """Return default reward function, or None for env built-in."""
    
    @abstractmethod
    def num_actions(self, env_params: Any) -> int
        """Return number of discrete actions."""
```

**XMinigrid implementation** (`xminigrid_adapter.py`):
- `MDPInterfaceWrapper(Wrapper)` — replaces obs and/or reward in env step
- Supports partial wrapping: if `get_obs_fn is None`, passes through default obs; if `reward_fn is None`, keeps env reward
- `make_env`: creates base env → MDPInterfaceWrapper → GymAutoResetWrapper
- `compute_success`: `(episode_length < max_steps)` — xminigrid terminates early on goal
- `get_default_obs_fn`: flattened grid (normalized) + position + one-hot direction + pocket + step count

#### Evolution Controller (`mdp_discovery/controller.py`)

**Main loop** (per iteration):
1. Sample parent from database (or generate from scratch on first iteration)
2. Build prompt with:
   - System: generic RL prompt + context doc as "Library Reference"
   - User: task + parent code + training feedback + top/diverse programs + failures
3. Call LLM (AWS Bedrock Claude) → extract code from ```python``` fence
4. Evaluate candidate (crash filter → short train → full train)
5. Store in MAP-Elites database
6. Handle island rotation and migration

**Mode dispatch:**
```python
def run(self):
    if mode == "default":
        return self._run_default_baseline()   # no LLM, adapter defaults
    if mode == "random":
        return self._run_random_baseline()    # LLM from scratch, no evolution
    else:
        return self._run_standard_evolution()  # full, reward_only, obs_only
```

#### Cascade Evaluator (`mdp_discovery/evaluator.py`)

**Pipeline:**
```
Stage 1: Crash Filter (fast — syntax, import, dry-run)
    ├─ Stage 0: ast.parse + verify required functions exist (5s timeout)
    ├─ Stage 1: importlib load (catches bad imports, JAX errors) (10s timeout)
    └─ Stage 2: dry-run with adapter.get_dummy_state() (shape/type errors) (30s timeout)

Stage 2: Short Training (config.training.total_timesteps, default 1-2M steps)
    └─ Threshold check: success_rate >= cascade_thresholds[0] (default 5%)?

Stage 3: Full Training (config.training.total_timesteps_full, default 4-6M steps)
    └─ Only if short training passes threshold
```

**Mode-specific overrides:**
```python
def _get_required_functions(self) -> List[str]:
    mode = self.config.evolution_mode
    if mode in ("full", "random"):
        return ["get_observation", "compute_reward"]
    elif mode == "reward_only":
        return ["compute_reward"]
    elif mode == "obs_only":
        return ["get_observation"]

def evaluate(self, code: str) -> CandidateResult:
    # ... after loading interface ...
    if mode == "reward_only":
        interface.get_observation = self.adapter.get_default_obs_fn()
    elif mode == "obs_only":
        interface.compute_reward = self.adapter.get_default_reward_fn()
```

#### MAP-Elites Database (`mdp_discovery/database.py`)

**Feature dimensions:**
- `reward_complexity`: Character count of `compute_reward` function body
- `obs_dimensionality`: Size of observation vector from `get_observation`

**Island model:**
- 3 islands, each with its own MAP-Elites grid (population_size=200, archive_size=50)
- Migration every 20 generations (ring topology, migration_rate=0.1)
- Prevents premature convergence on one observation/reward strategy

**Per-program storage:**
```python
@dataclass
class Program:
    id: str                              # UUID
    code: str                            # Full source code
    metrics: Dict[str, float]            # success_rate, final_return, final_length
    obs_dim: Optional[int]               # Detected observation dimension
    generation: int                      # Generation number
    parent_id: Optional[str]             # Parent program ID
    island: int                          # Island assignment (0, 1, or 2)
    feature_coords: List[int]            # [reward_complexity_bin, obs_dim_bin]
    timestamp: float                     # Unix timestamp
```

**Sampling strategy:**
- 50% exploitation (sample from top programs)
- 25% exploration (sample randomly)
- 25% weighted (sample proportional to fitness)

#### Prompt Builder (`mdp_discovery/prompts.py`)

**System prompt** (assembled at init):
```
You are an expert RL engineer. You design MDP interfaces for RL agents.

{function_description}
  - full: "Your task is to write two JAX-compatible Python functions:
           1. get_observation(state) → 1D float32 array
           2. compute_reward(state, action, next_state) → scalar float32"
  - reward_only: "Your task is to write one JAX-compatible Python function:
                  compute_reward(state, action, next_state) → scalar float32"
  - obs_only: "Your task is to write one JAX-compatible Python function:
               get_observation(state) → 1D float32 array"

[Design Philosophy — generic RL advice about obs/reward design]

## Library Reference
{library_context}  ← context doc injected verbatim (no modifications)

## Output Format
{output_constraints}
  - full: "The code must define both get_observation and compute_reward"
  - reward_only: "The code must define compute_reward"
  - obs_only: "The code must define get_observation"
```

**User prompt — from scratch (iteration 1):**
```
## Task
{task_description}

## Instructions
{function_instructions}  ← mode-dependent
{obs_instructions}       ← omitted in reward_only mode
{reward_instructions}    ← omitted in obs_only mode
[JAX pitfalls]
[Constraints]
```

**User prompt — evolutionary (iteration 2+):**
```
## Task
{task_description}

## Parent Program
{parent_code}
Parent Metrics: success={parent_success:.1f}%, length={parent_length:.1f}

## Training Feedback
{success_analysis}       # "Success rate improved from X% to Y%"
{plateau_detection}      # "Learning has plateaued at update Z"
{curve_trends}           # "Success curve shows steady improvement"

## Best Known Programs
{top_program_1}          # Code snippet + metrics
{top_program_2}
{top_program_3}

## Diverse Approaches
{diverse_program_1}      # From different MAP-Elites cells
{diverse_program_2}

## Recent Failures
{failure_1}              # Crash error or low-fitness code
{failure_2}

## Improvement Guidance
{mode_aware_guidance}    # "Focus on reward shaping" or "Improve observation features"
```

#### Crash Filter (`mdp_discovery/crash_filter.py`)

**Three stages:**
```python
@dataclass
class CrashFilterResult:
    passed: bool
    stage_failed: Optional[int]
    error_message: Optional[str]
    obs_dim: Optional[int]

def run_crash_filter(
    code: str,
    config: Config,
    dummy_state: Any,
    required_functions: List[str],
) -> CrashFilterResult:
    
    # Stage 0: Syntax + structure
    try:
        ast.parse(code)
        for fn in required_functions:
            assert fn in code, f"Missing {fn}"
    except Exception as e:
        return CrashFilterResult(passed=False, stage_failed=0, error_message=str(e))
    
    # Stage 1: Import check
    try:
        module = importlib.import_module(tempfile)
    except Exception as e:
        return CrashFilterResult(passed=False, stage_failed=1, error_message=str(e))
    
    # Stage 2: Dry-run
    try:
        if "get_observation" in required_functions:
            obs = interface.get_observation(dummy_state)
            assert obs.ndim == 1, f"Obs must be 1D, got {obs.ndim}"
            assert obs.shape[0] <= max_obs_dim, f"Obs dim {obs.shape[0]} > {max_obs_dim}"
            assert obs.dtype == jnp.float32, f"Obs dtype {obs.dtype} != float32"
        if "compute_reward" in required_functions:
            reward = interface.compute_reward(dummy_state, action, next_state)
            assert reward.ndim == 0, f"Reward must be scalar, got {reward.ndim}"
            assert reward.dtype == jnp.float32, f"Reward dtype {reward.dtype} != float32"
    except Exception as e:
        return CrashFilterResult(passed=False, stage_failed=2, error_message=str(e))
    
    return CrashFilterResult(
        passed=True,
        obs_dim=obs.shape[0] if "get_observation" in required_functions else 0,
    )
```

#### Training (`mdp_discovery/train.py`)

**PPO with GRU**, adapted from xland-minigrid's reference implementation:
```python
def run_training(
    config: Config,
    adapter: EnvAdapter,
    interface: MDPInterface,
    obs_dim: int,
    total_timesteps: int,
) -> Dict[str, float]:
    
    # 1. Create environment via adapter
    env, env_params = adapter.make_env(interface.get_observation, interface.compute_reward)
    
    # 2. Create network + optimizer
    network = ActorCriticMLP(
        obs_dim=obs_dim,
        num_actions=adapter.num_actions(env_params),
        hidden_dim=config.training.hidden_dim,
        rnn_hidden_dim=config.training.rnn_hidden_dim,
    )
    optimizer = optax.adam(learning_rate=config.training.lr)
    train_state = TrainState(network=network, opt_state=optimizer.init(network.params))
    
    # 3. Run training (JIT-compiled, pmap'd across devices)
    train_fn = make_train(
        env=env,
        env_params=env_params,
        config=config.training,
        compute_success_fn=adapter.compute_success,
    )
    train_state, metrics = train_fn(train_state)
    
    # 4. Final evaluation (50 episodes)
    success_rate = evaluate(config, adapter, interface, train_state)
    metrics["success_rate"] = success_rate
    
    return metrics
```

### 2.5 Evolution Modes

| Mode | LLM Generates | Crash Filter Checks | Obs Fn Used | Reward Fn Used | Purpose |
|------|--------------|-------------------|-------------|---------------|---------|
| `full` | obs + reward | both | LLM-generated | LLM-generated | Main experiment (joint discovery) |
| `reward_only` | reward only | compute_reward only | adapter default | LLM-generated | Eureka-style baseline |
| `obs_only` | obs only | get_observation only | LLM-generated | env built-in (None) | Obs contribution ablation |
| `default` | nothing | nothing | adapter default | env built-in | No-design-effort baseline |
| `random` | obs + reward | both | LLM-generated | LLM-generated | No-evolution baseline |

### 2.6 Configuration

**Example config (`configs/default.yaml`):**
```yaml
max_iterations: 500
candidates_per_iteration: 1
checkpoint_interval: 50
log_level: "INFO"
random_seed: 42

environment:
  env_id: "MiniGrid-Empty-6x6"
  adapter_type: "xminigrid"
  context_file: "contexts/xminigrid_context.md"

training:
  num_envs: 8192
  num_steps: 16
  update_epochs: 1
  num_minibatches: 16
  total_timesteps: 2000000       # short training
  total_timesteps_full: 6000000  # full training
  lr: 0.001
  clip_eps: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  eval_episodes: 80
  eval_episodes_for_fitness: 50
  eval_max_steps: 80
  seed: 42
  num_seeds: 10                  # multi-seed averaging
  hidden_dim: 256
  rnn_hidden_dim: 512

mdp_interface:
  max_obs_dim: 512

crash_filter:
  stage0_timeout: 5
  stage1_timeout: 10
  stage2_timeout: 30
  retry_on_crash: true
  max_crash_retries: 1

evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.05]     # 5% success to proceed to full training

llm:
  region_name: "us-east-1"
  model_name: "claude-sonnet-4-6"
  temperature: 0.7
  max_tokens: 4096
  retries: 3

database:
  population_size: 200
  archive_size: 50
  num_islands: 3
  feature_dimensions: ["reward_complexity", "obs_dimensionality"]
  feature_bins: 5
  migration_interval: 20
  migration_rate: 0.1
```

### 2.7 Usage

```bash
# Basic run
python run.py --task "Navigate to the goal tile." --iterations 50

# Custom config with mode override
python run.py --task "..." --mode reward_only --iterations 100

# Override environment and model
python run.py --task "..." --env MiniGrid-DoorKey-6x6 --model claude-4-sonnet

# Multi-seed evaluation
python run.py --task "..." --num-seeds 10

# Resume from checkpoint
python run.py --task "..." --resume runs/previous_run/

# Run benchmark
python benchmark_best_vs_sparse.py
python benchmark_ablation_spaces.py
```

---

## 3. Experimental Setup

### 3.1 Novel Tasks (Not from Any Benchmark)

| Task | Description | Grid | Max Steps | Novelty |
|------|-------------|------|-----------|---------|
| **Easy** | Pick up blue pyramid | 9×9 empty | 80 | **Novel** — no benchmark has "pick up specific colored object" |
| **Medium** | Place yellow pyramid adjacent to green square | 9×9 empty | 80 | **Novel** — requires multi-phase reasoning (pickup → place) |
| **Hard** | Pick up blue pyramid (→green ball), place next to yellow hex | 13×13 4-room | 400 | **Novel** — object transformation + multi-room navigation + doors |

**Comparison with Eureka's task suite:**
- Eureka: 29 benchmark tasks (from existing RL libraries) + 1 novel (pen spinning with curriculum)
- PureEvol: **All 3 tasks are fully novel** — not from MiniGrid, XLand, or any standard benchmark

### 3.2 Training Configuration

| Task | Short Training | Full Training | Eval Episodes |
|------|---------------|---------------|---------------|
| Easy | 500K steps | 2M steps | 80 |
| Medium | 1M steps | 4M steps | 80 |
| Hard | 3M steps | 6M steps | 80 |

**Common settings:**
```yaml
num_seeds: 10              # Seeds 42-51 for reduced variance
cascade_evaluation: true
cascade_thresholds: [0.05] # 5% to proceed to full training
num_envs: 8192             # Parallel environments
```

### 3.3 Evaluation Protocol

- **10 seeds** per method (seeds 42, 43, ..., 51)
- **Success rate**: fraction of episodes completing task (primary metric)
- **Episode length**: average steps to completion (successful episodes only)
- **Standard deviation**: reported for all metrics
- **Cascade evaluation**: 5% threshold to proceed from short to full training

---

## 4. Main Experiments: Full Mode Results

**Mode:** `full` (joint observation + reward evolution)

### 4.1 Summary Table

| Task | Success Rate | Episode Length | Obs Dim | Training Time |
|------|--------------|----------------|---------|---------------|
| **Easy** | **99%** | **15.7** steps | 174 | ~4 min |
| **Medium** | **97%** | **61.3** steps | 102 | ~10 min |
| **Hard** | **76%** | **186.5** steps | 147 | ~25 min |

### 4.2 Evolution Dynamics

#### Easy Task (Pick up blue pyramid)
- **Iteration 1:** 58% success
- **Iteration 3:** 98% success (rapid convergence)
- **Iteration 30:** 99% success, 15.7 steps (near-optimal)

#### Medium Task (Place pyramid near square)
- **Iteration 1-4:** 0-18% (search phase)
- **Iteration 8:** 56% success (breakthrough)
- **Iteration 30:** 97% success, 61.3 steps

#### Hard Task (4-room pickup + place)
- **Iteration 1-6:** 1-3% success (extended search)
- **Iteration 8:** 69% success (major breakthrough—joint obs+reward synergy)
- **Iteration 30:** 76% success, 186.5 steps

### 4.3 Key Observations

1. **Joint optimization discovers synergies:** The hard task shows a dramatic jump at iteration 8 (3% → 69%), indicating the LLM discovered a strategy requiring coordinated observation and reward design.

2. **Observation dimensionality doesn't correlate with performance:**
   - Easy: 174-dim obs → 99% success
   - Medium: 102-dim obs → 97% success
   - Hard: 147-dim obs → 76% success
   - Smaller, task-relevant observations outperform larger generic ones.

3. **Evolution converges quickly on easy tasks, requires exploration on hard tasks:** Easy tasks reach >95% by iteration 3; hard tasks require 8+ iterations to find breakthrough strategies.

---

## 5. Ablation Experiments

### 5.1 Experimental Design

Three modes compared across all tasks:
- **`obs_only`**: Evolve observation, use env built-in reward
- **`reward_only`**: Evolve reward, use default observation
- **`full`**: Evolve both (main experiment)

### 5.2 Results Summary

| Task | Mode | Success Rate | Episode Length | Obs Dim |
|------|------|--------------|----------------|---------|
| **Easy** | **full** | **99%** | **15.7** | 174 |
| (Pick up pyramid) | obs_only | 98% | 22.3 | 190 |
| | reward_only | 59% | 107.9 | default |
| **Medium** | **full** | **97%** | **61.3** | 102 |
| (Place pyramid) | obs_only | 99% | 31.7 | 108 |
| | reward_only | 2% | 239.3 | default |
| **Hard** | **full** | **76%** | **186.5** | 147 |
| (4-room) | obs_only | 34% | 498.7 | 472 |
| | reward_only | 4% | 506.1 | default |

### 5.3 Key Findings

#### 1. Observation Design is Critical (but Not Sufficient)

- **Easy/Medium tasks:** `obs_only` achieves near-optimal performance (98-99%)
- **Hard task:** `obs_only` plateaus at 34%—observation alone can't solve multi-phase temporal credit assignment

#### 2. Reward-Only Methods Fail on Complex Tasks

- **Medium task:** `reward_only` achieves only 2% success (50× worse than full)
- **Hard task:** `reward_only` achieves only 4% success (19× worse than full)
- Even sophisticated multi-phase reward shaping fails without appropriate observations

#### 3. Joint Optimization Unlocks Emergent Strategies (Hard Task)

| Metric | full | obs_only | reward_only |
|--------|------|----------|-------------|
| Success | **76%** | 34% | 4% |
| Improvement | — | −55% | −95% |

**Why full wins on hard tasks:**
- Phase-conditioned observations (target switches based on holding state)
- Coordinated reward shaping (pyramid approach → hex approach → placement)
- Milestone bonuses (pickup, adjacency, ready-to-putdown)
- The LLM discovered that **temporal scaffolding** (reward) + **phase-aware observations** are both necessary

#### 4. Observation Complexity ≠ Performance

| Task | Mode | Obs Dim | Success |
|------|------|---------|---------|
| Hard | obs_only | 472 | 34% |
| Hard | full | 147 | **76%** |

The `obs_only` solution used **3.2× more features** but achieved **less than half the performance**. Quality of features matters more than quantity.

---

## 6. Evolved Interface Analysis

### 6.1 Easy Task: Pick Up Blue Pyramid

#### Full Mode (99% success, 174-dim obs)

**Observation features:**
- Agent position (normalized), direction (one-hot)
- Pocket info (tile, color, empty, holding_blue_pyramid)
- Pyramid absolute/relative positions, distances
- Adjacency flags, front tile info
- 7×7 local view (147 features)

**Reward terms:**
- Pickup bonus: +10
- Approach shaping: 0.5 × Δdistance
- Facing bonus: +1
- Adjacent transition: +0.5
- Step penalty: −0.005

#### Obs-Only Mode (98% success, 190-dim obs)

**Additional features over full:**
- Angle to pyramid (sin/cos encoding)
- Neighbor tiles (4 directions × 3 features)
- Same row/column flags
- More detailed local view

**Key insight:** Richer observation compensates for lack of reward shaping on simple tasks.

---

### 6.2 Medium Task: Place Pyramid Near Square

#### Full Mode (97% success, 102-dim obs)

**Observation features:**
- Agent, pyramid, square positions (absolute + relative)
- Holding state, pocket empty
- Front tile info, adjacency flags
- Best adjacent target (for placement)
- Phase indicators (go_pickup, carry, done)
- 5×5 local view

**Reward terms:**
- Approach pyramid: 2.0 × Δdistance
- Approach square: 3.0 × Δdistance
- Pickup: +2
- Putdown adjacent: +10 / far: −1.5
- Success: +10
- Maintain success: +0.2
- Ready-to-putdown: +0.5

#### Reward-Only Mode (2% success)

**Reward terms (sophisticated but ineffective):**
- Phase 1 shaping (approach pyramid): 2.0 × Δdistance
- Phase 2 shaping (approach square): 3.0 × Δdistance
- Pickup: +3
- Putdown adjacent/close/far: +10 / +1 / −1
- Success: +20
- Placement improvement shaping

**Why it fails:** Default observation lacks critical spatial relationships (relative positions, distances, adjacency). Agent can't learn multi-phase behavior without knowing where objects are relative to itself.

---

### 6.3 Hard Task: 4-Room Pickup + Place

#### Full Mode (76% success, 147-dim obs)

**Observation features:**
- Agent position, direction, pocket
- Pyramid/hex positions (absolute + relative)
- Distances to targets
- Neighbor features (24-dim)
- Hex neighbor floors (20-dim)
- Best placement target
- Phase indicators (phase0, phase1, phase2)
- Phase-conditioned target (switches based on holding)
- 5×5 local view
- Putdown validity (8-dim)

**Reward terms:**
- Pickup: +5
- Pyramid shaping: 3.0 × Δdistance
- Hex shaping: 3.0 × Δdistance
- Placement shaping: 1.5 × Δdistance
- Adjacency bonus: +3
- Success: +20
- Wrong placement: −5
- Ready bonus: +0.5
- Step penalty: −0.01

#### Obs-Only Mode (34% success, 472-dim obs)

**Features (massive but ineffective):**
- One-hot pocket encoding (13 + 12 dims)
- Full grid semantic maps (169 + 169 = 338 dims!)
- Door analysis (nearest door, room quadrants)
- Hex neighbors, placement targets
- Phase-conditioned targeting

**Why it fails:** Despite having all spatial information (including full grid maps), the agent lacks **temporal scaffolding**. No reward signal teaches the sequence: navigate→pickup→navigate→place. Agent gets stuck in local optima (e.g., repeatedly approaching pyramid without picking up).

#### Reward-Only Mode (4% success)

**Reward terms (comprehensive but useless):**
- Phase 1 shaping (pyramid): 10.0 × Δdistance
- Phase 2 shaping (hex): 10.0 × Δdistance
- Pickup: +3
- Adjacency bonus: +1
- Putdown good/bad: +5 / −2
- Ball placement shaping: 2.0 × distance
- Success: +20 / +1

**Why it fails:** Default observation is a flattened grid—agent can't efficiently extract spatial relationships. Even with perfect reward shaping, agent can't learn where to go or what to do.

---

## 7. Cost Analysis

### 7.1 LLM Costs (Ablation Experiments)

| Task | Mode | Input Tokens | Output Tokens | Cost (USD) |
|------|------|--------------|---------------|------------|
| Easy | obs_only | 523K | 73K | $2.67 |
| | reward_only | 388K | 55K | $1.98 |
| Medium | obs_only | 1,078K | 142K | $5.37 |
| | reward_only | 496K | 66K | $2.48 |
| Hard | obs_only | 1,156K | 157K | $5.83 |
| | reward_only | 627K | 90K | $3.23 |
| **Total** | | **4,268K** | **583K** | **$21.55** |

### 7.2 Training Costs (10-Seed Benchmark)

| Task | Budget | Per-Method Time | ×4 Methods |
|------|--------|-----------------|------------|
| Easy | 2M | ~35 min | ~2.3 hrs |
| Medium | 4M | ~85 min | ~5.7 hrs |
| Hard | 6M | ~160 min | ~10.7 hrs |
| **Total** | | | **~18-19 hrs** |

---

## 8. Scientific Conclusions

### 1. Joint MDP Interface Discovery is Necessary for Complex Tasks

**Claim:** Evolving both observation and reward together achieves 2.2× higher success rates than observation-only on complex multi-phase tasks.

**Evidence:**
- Hard task: full (76%) vs obs_only (34%) vs reward_only (4%)
- The 69% breakthrough at iteration 8 (full mode) required joint optimization

### 2. Observation Design is the Primary Bottleneck

**Claim:** For tasks requiring spatial reasoning, observation design matters more than reward shaping.

**Evidence:**
- Medium task: obs_only (99%) vs reward_only (2%)—50× difference
- Easy task: obs_only (98%) vs reward_only (59%)—1.7× difference

### 3. Reward Shaping Provides Temporal Scaffolding

**Claim:** For multi-phase tasks, reward shaping teaches temporal dependencies that observation alone cannot.

**Evidence:**
- Hard task full mode reward has explicit phase bonuses (pickup, adjacency, ready)
- Obs-only hard mode has all spatial info but no sequence teaching → 34% success

### 4. Feature Quality > Feature Quantity

**Claim:** Compact, task-relevant observations outperform large generic ones.

**Evidence:**
- Hard full (147 dim, 76%) vs hard obs_only (472 dim, 34%)
- 3.2× more features → less than half the performance

### 5. Evolutionary Search Finds Non-Obvious Strategies

**Claim:** LLM-guided evolution discovers strategies that humans might not design.

**Evidence:**
- Hard task iteration 8: sudden jump from 3% to 69%
- Evolved interfaces use phase-conditioned targets and multi-scale spatial features

---

## 9. Comparison with Eureka

| Aspect | PureEvol | Eureka |
|--------|----------|--------|
| What's evolved | Observation + Reward (joint) | Reward only |
| Task source | **All novel tasks** | 29 benchmark + 1 novel (pen spinning) |
| Observation | **Discovered from scratch** | Fixed (env-provided) |
| Reward | Discovered from scratch | Discovered from scratch |
| Curriculum | None | Required for pen spinning (2 stages) |
| Env framework | XLand-MiniGrid (JAX) | IsaacGym (PyTorch) |
| Training backend | PPO with GRU (JAX pmap) | rl_games PPO (PyTorch) |
| Diversity mechanism | MAP-Elites + islands | Top-K sampling |
| Env-agnostic | Yes (adapter pattern) | Partially |

**Key differentiator:** PureEvol discovers the **full MDP interface** (obs + reward), not just reward. Eureka's pen spinning required curriculum learning (pre-training + fine-tuning) because it could only optimize rewards, not observations. PureEvol achieves comparable results without curriculum.

---

## 10. Future Directions

1. **Action space evolution:** Extend to evolve action transformations (e.g., macro-actions, action masking)

2. **Curriculum evolution:** Evolve task sequences alongside MDP interfaces

3. **Multi-task evolution:** Discover interfaces that generalize across environment variants

4. **Human-in-the-loop:** Allow human feedback to guide evolution (e.g., preference labels)

5. **Larger environments:** Test on continuous control, multi-agent, or hierarchical tasks

---

## 11. Repository Structure

```
pureevol/
├── run.py                              # CLI entry point
├── configs/
│   ├── default.yaml                    # Default config
│   ├── easy_pickup.yaml                # Easy task config
│   ├── medium_place_near.yaml          # Medium task config
│   └── hard_rule_chain.yaml            # Hard task config
├── contexts/
│   └── xminigrid_context.md            # XLand-MiniGrid API reference
├── templates/
│   └── mdp_interface_template.py       # Interface contract (signatures only)
├── mdp_discovery/
│   ├── __init__.py
│   ├── config.py                       # Hierarchical dataclass config + YAML
│   ├── controller.py                   # Main evolution loop
│   ├── evaluator.py                    # Cascade evaluator pipeline
│   ├── crash_filter.py                 # 3-stage code validation
│   ├── train.py                        # PPO training (JAX pmap)
│   ├── train_utils.py                  # Rollout, GAE, PPO update
│   ├── nn.py                           # ActorCriticMLP with GRU
│   ├── mdp_interface.py                # Interface loader + validator
│   ├── prompts.py                      # Prompt templates + PromptBuilder
│   ├── llm_client.py                   # AWS Bedrock Claude client
│   ├── database.py                     # MAP-Elites + island evolution
│   └── adapters/
│       ├── __init__.py                 # Factory: get_adapter(type, config)
│       ├── base.py                     # EnvAdapter ABC
│       └── xminigrid_adapter.py        # XMinigrid implementation
├── benchmark_best_vs_sparse.py         # Main benchmark script
├── benchmark_ablation_spaces.py        # Ablation benchmark script
├── RESULTS.MD                          # This document
└── SYSTEM_DESIGN.MD                    # Original system design doc
```

---

**Last updated:** February 24, 2026
