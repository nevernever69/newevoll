# System Architecture: MDP Interface Discovery

This document describes the complete architecture of the MDP Interface Discovery system — an LLM-guided evolutionary search engine that discovers observation mappings and reward functions for reinforcement learning tasks.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Entry Point and Configuration](#2-entry-point-and-configuration)
3. [The Evolution Controller](#3-the-evolution-controller)
4. [LLM Code Generation](#4-llm-code-generation)
5. [Prompt Construction](#5-prompt-construction)
6. [Three-Stage Crash Filter](#6-three-stage-crash-filter)
7. [Cascade Evaluation Pipeline](#7-cascade-evaluation-pipeline)
8. [MAP-Elites Database with Island Model](#8-map-elites-database-with-island-model)
9. [Environment Adapters](#9-environment-adapters)
10. [MDP Interface Wrapper](#10-mdp-interface-wrapper)
11. [Training Backends](#11-training-backends)
12. [Discovery Task Wrappers](#12-discovery-task-wrappers)
13. [Evolution Tracing](#13-evolution-tracing)
14. [Checkpointing and Resume](#14-checkpointing-and-resume)
15. [Evolution Modes and Ablations](#15-evolution-modes-and-ablations)
16. [Complete Data Flow](#16-complete-data-flow)
17. [File Map](#17-file-map)

---

## 1. High-Level Overview

The system solves a bilevel optimization problem: the outer loop uses an LLM to generate candidate MDP interfaces (observation + reward functions as Python code), and the inner loop trains RL agents with each candidate to measure task success.

```
                    ┌─────────────────────────────────────────────┐
                    │              OUTER LOOP                      │
                    │                                             │
 Task Description ─>│  ┌─────────┐   ┌──────────┐   ┌─────────┐ │
 Context Document ─>│  │   LLM   │──>│  Crash   │──>│Cascade  │ │
 Config YAML ──────>│  │Generate │   │  Filter  │   │  Train  │ │
                    │  └────▲────┘   │(3 stage) │   │  (PPO)  │ │
                    │       │        └──────────┘   └────┬────┘ │
                    │       │                            │      │
                    │       │   ┌──────────────────┐     │      │
                    │       └───│   MAP-Elites DB  │<────┘      │
                    │           │  (Island Model)  │            │
                    │           └──────────────────┘            │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                              Trained RL Agent +
                         Evolved Obs & Reward Code
```

**Three inputs, one output:**
- **Inputs**: (1) Natural language task description, (2) Context document (environment API reference, ~2 pages markdown), (3) YAML config file
- **Output**: Trained RL agent with evolved `get_observation(state)` and `compute_reward(state, action, next_state)` functions

**No hand-designed reward or observation is required.**

---

## 2. Entry Point and Configuration

### `run.py` — CLI Entry Point

Parses command-line arguments, loads YAML config, applies overrides, creates the `EvolutionController`, and runs the loop.

**Key CLI arguments:**
| Flag | Purpose |
|------|---------|
| `--config` | Path to YAML config file |
| `--task` | Natural language task description (overrides config) |
| `--mode` | Evolution mode: `full`, `reward_only`, `obs_only`, `default`, `random` |
| `--iterations` | Max evolution iterations |
| `--timesteps` | Short training timesteps |
| `--timesteps-full` | Full training timesteps |
| `--num-seeds` | Seeds for multi-seed averaging |
| `--no-cascade` | Disable cascade, go straight to full training |
| `--resume` | Resume from checkpoint directory |
| `--run-dir` | Explicit output directory |
| `--model` | LLM model (supports ensemble: `"model1:w1,model2:w2"`) |

**Logging**: Dual-tier — DEBUG to file (`evolution.log`), INFO+ to console. Noisy libraries (boto3, jax, mujoco, etc.) suppressed to WARNING.

### `config.py` — Hierarchical Configuration

Dataclass-based config system loaded from YAML via `dacite`. All config fields have sensible defaults.

**Config hierarchy:**

```
Config (root)
├── max_iterations: 500
├── candidates_per_iteration: 1
├── evolution_mode: "full"
├── checkpoint_interval: 50
├── random_seed: 42
│
├── EnvironmentConfig
│   ├── env_id: "MiniGrid-Empty-6x6"
│   ├── adapter_type: "xminigrid" | "mujoco"
│   └── context_file: path to markdown
│
├── TrainingConfig
│   ├── total_timesteps: 1_000_000        (short training)
│   ├── total_timesteps_full: 5_000_000   (full training)
│   ├── num_seeds: 3
│   ├── num_envs: 8192
│   ├── lr: 0.001, gamma: 0.99, etc.
│   └── brax_*: MuJoCo-specific PPO settings
│
├── CrashFilterConfig
│   ├── stage0_timeout: 5s
│   ├── stage1_timeout: 10s
│   └── stage2_timeout: 30s
│
├── EvaluatorConfig
│   ├── cascade_evaluation: true
│   ├── cascade_thresholds: [0.05]
│   └── success_threshold: 0.02
│
├── LLMConfig
│   ├── model_name: "claude-sonnet-4-6"
│   ├── temperature: 0.7
│   ├── region_name: "us-east-1"
│   └── models: [...] (for ensemble)
│
├── DatabaseConfig
│   ├── num_islands: 3
│   ├── feature_dimensions: ["reward_complexity", "obs_dimensionality"]
│   ├── feature_bins: 5
│   ├── migration_interval: 20
│   └── exploitation_ratio: 0.7, exploration_ratio: 0.2
│
├── PromptConfig
│   ├── num_top_programs: 3
│   ├── num_diverse_programs: 2
│   ├── num_failure_examples: 2
│   └── use_stochasticity: true
│
└── EvolutionTraceConfig
    ├── enabled: true
    └── include_code: false
```

---

## 3. The Evolution Controller

**File:** `mdp_discovery/controller.py`

The `EvolutionController` orchestrates the entire evolution loop. It owns the database, evaluator, LLM client, prompt builder, and evolution tracer.

### Initialization

```python
controller = EvolutionController(
    config=config,
    task_description="Pick up the blue pyramid.",
    checkpoint_dir="runs/easy_full_20260301",
)
```

On construction:
1. Creates the environment adapter (`XMinigridAdapter` or `MujocoAdapter`)
2. Loads the context document (markdown file describing the environment API)
3. Initializes the `ProgramDatabase` (MAP-Elites + islands)
4. Creates the `CascadeEvaluator`
5. Initializes the `LLMClient` (or `LLMEnsemble`)
6. Creates the `PromptBuilder` with context and task description
7. Sets up the `EvolutionTracer` for logging

### Main Loop

```python
controller.run()
```

**Mode dispatch:**
- `"default"` → `_run_default_baseline()` — no LLM, trains with adapter defaults once
- `"random"` → `_run_random_baseline()` — LLM generates from scratch each time, no parent feedback
- `"full"`, `"reward_only"`, `"obs_only"` → standard evolutionary loop

**Standard loop (per iteration):**

```
1. PREPARE (main thread):
   - Sample parent from database (or None for first iteration)
   - Retrieve top programs, diverse programs, failures, inspiration
   - Build prompt via PromptBuilder

2. GENERATE (worker thread):
   - Call LLM with prompt → get response text
   - Extract Python code from response
   - If no code extracted: retry once

3. EVALUATE (worker thread):
   - Run 3-stage crash filter
   - If crashed + retry enabled: feed error to LLM, retry once
   - If passed: cascade training (short → threshold → full)

4. STORE (main thread):
   - Add result to database (MAP-Elites placement)
   - Log iteration stats
   - Manage island rotation and migration
   - Checkpoint periodically
```

### Island Rotation and Migration

The controller rotates between islands every iteration (round-robin via `db.next_island()`). When `max(island_generations) - last_migration_gen >= migration_interval`, it triggers inter-island migration — top programs from each island migrate to the next in a ring topology.

---

## 4. LLM Code Generation

**File:** `mdp_discovery/llm_client.py`

### LLM Client

Uses AWS Bedrock's Converse API. Supports multiple model families:

| Short Name | Model |
|------------|-------|
| `claude-4-sonnet` | Claude Sonnet 4.6 |
| `claude-4-opus` | Claude Opus 4.6 |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet |
| `llama4-maverick` | Llama 4 Maverick |
| `mistral-large` | Mistral Large |

**Thread safety:** Uses thread-local Bedrock clients (one per worker thread).

**Ensemble mode:** When `config.llm.models` is set, `LLMEnsemble` wraps multiple models with weighted random selection per generation.

### Generation Flow

```python
response = llm.generate({"system": system_prompt, "user": user_prompt})
# response.text: full LLM output
# response.code: extracted Python code (or None)
# response.input_tokens, response.output_tokens: for cost tracking
```

Code extraction uses three-tier regex:
1. `` ```python ... ``` `` fence (preferred)
2. Any `` ``` ... ``` `` fence containing Python-like code
3. Bare code starting with `import jax` (fallback for when LLM forgets fence)

### Retry Logic

If `response.code is None` (no code extracted), the controller retries once with an explicit "please provide code in a Python fence" message.

If the code crashes during evaluation and `retry_on_crash` is enabled, the controller sends the crash error message back to the LLM and gets a corrected version.

---

## 5. Prompt Construction

**File:** `mdp_discovery/prompts.py`

The `PromptBuilder` constructs system/user message pairs tailored to the current evolution state.

### System Prompt

A generic RL engineering prompt with **zero environment-specific vocabulary**. Contains:
- Role description ("You are an RL engineer designing MDP interfaces")
- Mode-specific function descriptions (which functions to write)
- Design philosophy (help the network learn, don't solve the task)
- Observation guidelines (normalize, include all needed info, max 512 dims)
- Reward guidelines (guide toward task, evaluated on success rate not magnitude)
- Generalization warnings (randomized env, don't hardcode)
- **Library Reference**: The full context document injected verbatim
- Output format constraints (JAX-only, function signatures)

### User Prompt: Two Templates

**From-scratch** (first iteration or new island with no parent):
- Task description
- Recent failures (if any) — with error messages for learning
- Instructions to write both functions

**Evolutionary** (has parent code):
- Parent code with metrics (success rate, reward, obs_dim)
- Training feedback (per-seed success, learning curves, plateau detection)
- Top programs from the population (for reference)
- Diverse programs from different MAP-Elites cells
- Inspiration programs (structurally distant, for exploration)
- Recent failures (crashes and low-fitness programs)
- Improvement guidance (mode-specific, success-bracket-specific)

### Improvement Guidance System

The prompt builder selects guidance based on the parent's performance bracket:

| Success Bracket | Guidance Strategies (3 "lenses", one selected randomly) |
|----------------|--------------------------------------------------------|
| **Zero** (0%) | A: Focus on observation basics. B: Focus on reward signal. C: Try structural reset. |
| **Low** (< 30%) | A: What info is missing? B: Is reward signal too weak? C: Are obs and reward coherent? |
| **Medium** (30–60%) | A: Analyze failure modes. B: Simplify. C: Enrich with new features. |
| **Good** (60–90%) | A: Target remaining failures. B: Compress observation. C: Tune reward weights. |
| **Excellent** (> 90%) | A: Targeted edge-case fixes. B: Minimize obs dim. C: Robustness testing. |

**Stochasticity**: When `use_stochasticity=True`, a random lens is selected per iteration. This prevents the LLM from getting stuck in one optimization direction.

**Plateau detection**: If the parent's fitness ≈ population best and best < 85%, the prompt includes a strong signal: "The population may be stuck. Consider a structurally different approach."

### Training Feedback Formatting

The prompt includes detailed analysis of the parent's training run:
- Per-seed success rates and spread (consistency indicator)
- NaN detection warnings
- Performance bracket analysis
- Plateau detection (near best, stagnating)
- Reward curve trend (increasing? declining? flat?)
- Success curve trend (improving? declining? stuck at zero from start?)

---

## 6. Three-Stage Crash Filter

**File:** `mdp_discovery/crash_filter.py`

LLM-generated code frequently contains errors. The crash filter catches them before wasting expensive training compute. In practice, **30–50% of candidates crash** at some stage.

### Stage 0: Syntax Check (5s timeout)

```
Input: raw code string
Check: ast.parse() succeeds
Check: required function names exist in AST
Catches: SyntaxError, missing functions
```

### Stage 1: Import Check (10s timeout)

```
Input: parsed code
Check: importlib loads module successfully
Check: required functions are callable attributes
Catches: ImportError, AttributeError, numpy-vs-jax mistakes
```

### Stage 2: Dry-Run Validation (30–60s timeout)

```
Input: loaded module + real environment state from adapter.get_dummy_state()
Check: get_observation(state) returns 1D array, shape[0] <= 512
Check: compute_reward(state, action, state) returns scalar
Check: jax.jit(get_observation)(state) compiles without error
Check: jax.jit(compute_reward)(state, action, state) compiles without error
Check: no NaN/Inf in outputs
Catches: shape errors, Python control flow on JAX arrays, attribute errors on state
```

The JIT compilation check in Stage 2 is critical — it catches the most common LLM mistake: using Python `if` statements on JAX-traced values (e.g., `if state.info["done"] > 0.5:`), which silently produces wrong behavior or crashes during training.

### Error Feedback

Error messages and tracebacks from each stage are captured and included in subsequent LLM prompts. The crash stage name is human-readable (e.g., "Dry-Run (stage 2) — runtime error when calling functions with a real environment state").

### Output

```python
@dataclass
class CrashFilterResult:
    passed: bool                          # True if all 3 stages passed
    stage_failed: Optional[int]           # 0, 1, or 2
    error_message: Optional[str]          # Human-readable error
    error_traceback: Optional[str]        # Full traceback
    obs_dim: Optional[int]                # Detected observation size (if passed)
```

---

## 7. Cascade Evaluation Pipeline

**File:** `mdp_discovery/evaluator.py`

After passing the crash filter, candidates enter training evaluation. The cascade avoids spending expensive full training on bad candidates.

### Pipeline

```
Candidate Code
     │
     ▼
┌──────────────┐
│ Crash Filter │──── CRASHED → store failure, return
│  (3 stages)  │
└──────┬───────┘
       │ passed
       ▼
┌──────────────────┐
│  Short Training  │──── 0.5–6M steps, single seed
│  (fast screen)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Threshold Check  │──── success < 5%? → REJECTED, return
│                  │     (stores metrics but doesn't promote)
└──────┬───────────┘
       │ passed
       ▼
┌──────────────────┐
│  Full Training   │──── 1–30M steps, 3 seeds
│  (multi-seed)    │──── averaged success rate = fitness
└──────┬───────────┘
       │
       ▼
  CandidateResult
  (fitness, metrics, obs_dim)
```

### Evaluation Stages

| Stage | Description | Typical Cost |
|-------|-------------|-------------|
| `CRASHED` | Failed crash filter | < 1 min |
| `SHORT_TRAIN` | Passed short training | 1–5 min |
| `SHORT_TRAIN_REJECTED` | Below cascade threshold | 1–5 min |
| `FULL_TRAIN` | Completed full multi-seed training | 5–30 min |

### Mode-Specific Overrides

Before training, the evaluator applies mode-specific substitutions:
- **reward_only**: Replace `get_observation` with `adapter.get_default_obs_fn()`
- **obs_only**: Set `compute_reward = None` (use environment built-in reward)
- **full**: Use both LLM-generated functions as-is

### NaN Handling

Both short and full training check for NaN in metrics. If NaN is detected:
- Short training: candidate is immediately rejected
- Full training: falls back to short training metrics (if available)

### CandidateResult

```python
@dataclass
class CandidateResult:
    code: str
    stage: EvalStage
    crash_result: CrashFilterResult
    metrics: Optional[Dict]          # success_rate, final_return, curves, per_seed, ...
    obs_dim: Optional[int]
    eval_time: float

    @property
    def fitness(self) -> float:
        return metrics.get("success_rate", 0.0)
```

---

## 8. MAP-Elites Database with Island Model

**File:** `mdp_discovery/database.py`

The database maintains the evolutionary population using MAP-Elites for quality-diversity and an island model for additional diversity.

### Data Structures

**Program** — A single evaluated MDP interface:
```python
@dataclass
class Program:
    id: str                    # UUID
    code: str                  # Python source
    metrics: Dict[str, float]  # {success_rate, final_return, per_seed_success, ...}
    obs_dim: int               # Observation dimensionality
    stage: str                 # Evaluation stage reached
    generation: int            # Born in which generation
    parent_id: Optional[str]   # Parent program UUID
    island: int                # Which island it belongs to
    timestamp: float
    feature_coords: List[int]  # [x, y] grid coordinates

    @property
    def fitness(self) -> float:
        return self.metrics.get("success_rate", 0.0)
```

**FailedProgram** — Crashed or zero-fitness candidate (kept for prompt feedback):
```python
@dataclass
class FailedProgram:
    code: str
    crash_result: Optional[CrashFilterResult]
    metrics: Optional[Dict]
    timestamp: float
```

### MAP-Elites Grid

Each island maintains a 2D grid indexed by two behavioral features:

| Dimension | Measurement | Purpose |
|-----------|-------------|---------|
| **Reward complexity** | AST node count in `compute_reward` | Structural diversity |
| **Observation dimensionality** | `obs_dim` (vector size) | Size diversity |

The grid is `feature_bins × feature_bins` (default 5×5 = 25 cells per island). Each cell stores the single highest-fitness program with those characteristics.

**Placement rule:** When a new program maps to a cell:
- Empty cell → place unconditionally
- Occupied cell → replace only if new fitness > current occupant's fitness

**Feature binning:** Min-max normalization based on observed feature ranges, then discretization into `feature_bins` bins. Feature stats (min/max) update adaptively as new programs arrive.

### Island Model

The system maintains `num_islands` (default 2–3) independent subpopulations, each with its own MAP-Elites grid.

**Rotation:** The controller cycles through islands round-robin. Each iteration generates a candidate for the current island.

**Migration** (ring topology):
- Triggered every `migration_interval` generations (default 5–20)
- Each island sends its top `migration_rate` fraction (default 10–20%) of programs to the next island in the ring
- Migrated programs are placed in the destination grid using MAP-Elites rules
- Prevents all islands from converging to the same local optimum

### Parent Selection

Three sampling strategies mixed by ratio:

| Strategy | Ratio | Method |
|----------|-------|--------|
| **Exploitation** | 70% | Random from global archive (elite programs) |
| **Exploration** | 20% | Uniform random from current island |
| **Weighted** | 10% | Fitness-proportional from current island |

Fallback: if the selected pool is empty, sample from any available program.

### Retrieval for Prompts

The database provides several views for the prompt builder:

| Method | Returns | Used For |
|--------|---------|----------|
| `get_best_program()` | Global best | Reference point |
| `get_top_programs(n)` | Top N by fitness | Exploitation examples in prompt |
| `get_diverse_programs(n)` | Programs from different cells | Structural variety |
| `get_inspiration_programs(n, parent_coords)` | Programs maximally distant from parent | Exploration nudges |
| `get_recent_failures(n)` | Last N crashes/low-fitness | Error feedback |

**Inspiration programs** are selected by Manhattan distance in feature space from the parent's coordinates — ensuring the LLM sees structurally different approaches (e.g., a 40-dim simple reward when the parent is 150-dim complex reward).

---

## 9. Environment Adapters

**Files:** `mdp_discovery/adapters/base.py`, `mdp_discovery/adapters/mujoco_adapter.py`

The adapter pattern isolates all environment-specific code. The core system (controller, prompts, crash filter, evaluator, database) never imports environment-specific modules.

### Abstract Base: `EnvAdapter`

```python
class EnvAdapter(ABC):
    # Required implementations:
    def make_env(self, get_obs_fn, reward_fn)           # Create training env
    def make_eval_env(self, get_obs_fn, reward_fn, max_steps)  # Create eval env
    def get_dummy_state(self)                             # Real state for crash filter
    def compute_success(self, rollout_stats, params)     # Success metric
    def get_default_obs_fn(self)                          # Default observation (for ablations)
    def get_default_reward_fn(self)                       # Default reward (for ablations)
    def num_actions(self, params)                         # Action space size

    # Optional overrides:
    def get_dummy_action(self)         # Default: jnp.int32(0)
    def is_continuous_action(self)     # Default: False
    def run_training(...)              # Default: calls train.py
    def run_training_multi_seed(...)   # Default: loops single-seed
```

### To Add a New Environment

1. **Implement `EnvAdapter`** — one file, ~100 lines
2. **Write a context document** — markdown describing state structure, action space, physical constants (~2 pages)
3. **Write a task wrapper** — thin subclass enriching `state.info` with pre-computed features for the LLM
4. **Create a config YAML** — training hyperparameters, timesteps, thresholds
5. **No changes to core system code required**

### MuJoCo Adapter

The `MujocoAdapter` uses a task registry to lazy-load environment classes:

```python
_TASK_REGISTRY = {
    "PandaPickAndTrack": "mdp_discovery.tasks.panda_pick_and_track.PandaPickAndTrack",
    "Go1PushRecovery": "mdp_discovery.tasks.go1_push_recovery.Go1PushRecovery",
}
```

Key differences from the XMinigrid adapter:
- Continuous action space (`is_continuous_action() = True`)
- Uses `MjxMDPInterfaceWrapper` to intercept `state.obs` and `state.reward`
- Overrides `run_training()` and `run_training_multi_seed()` to use Brax PPO backend
- Dummy action is `jnp.zeros(action_size)` instead of `jnp.int32(0)`

---

## 10. MDP Interface Wrapper

**File:** `mdp_discovery/mdp_interface.py`

The `MDPInterface` class loads and validates evolved code, then provides callable functions for the adapter.

### Loading

Two construction paths:

```python
# From file (checkpoint resume)
interface = MDPInterface.from_file("best_interface.py", required_functions=["get_observation", "compute_reward"])

# From code string (LLM output)
interface = MDPInterface.from_code(code_string, required_functions=["get_observation", "compute_reward"])
```

Both use `importlib` to dynamically load the code as a Python module.

### Validation

```python
obs_dim = interface.validate(dummy_state, max_obs_dim=512, dummy_action=zero_action)
```

Checks:
- `get_observation(state)` returns a 1D float array with `shape[0] <= 512`
- `compute_reward(state, action, state)` returns a scalar float
- No NaN/Inf in outputs

### MuJoCo Wrapper: `MjxMDPInterfaceWrapper`

Transparently wraps a MuJoCo environment, replacing observation and reward:

```python
class MjxMDPInterfaceWrapper:
    def reset(self, rng):
        state = self._env.reset(rng)
        if self._get_obs_fn:
            state = state.replace(obs={"state": self._get_obs_fn(state)})
        return state

    def step(self, state, action):
        prev = state
        state = self._env.step(state, action)
        if self._get_obs_fn:
            state = state.replace(obs={"state": self._get_obs_fn(state)})
        if self._reward_fn:
            state = state.replace(reward=self._reward_fn(prev, action, state))
        return state
```

The wrapper is transparent to Brax's training pipeline — it produces the same `State` dataclass with modified `obs` and `reward` fields.

---

## 11. Training Backends

### XLand-MiniGrid: `train.py`

PPO with GRU for discrete action spaces.

- **Network**: MLP encoder (256 hidden) → GRU (512 hidden) → Actor head + Critic head
- **Training**: `pmap`'d across devices, vectorized over `num_envs` (8192)
- **Returns**: `{success_rate, final_return, success_curve, learning_curve, training_time}`

### MuJoCo/Brax: `train_brax.py`

PPO with MLP for continuous action spaces (via Brax).

- **Network**: Separate policy (4×32) and value (5×256) MLPs
- **Training**: Brax's built-in `ppo.train()` with `wrap_for_brax_training`

**Multi-seed optimization**: The expensive XLA compilation (`train_fn` build) happens only once. Subsequent seeds reuse the compiled function — typically 3× faster than recompiling per seed.

```python
def run_training_multi_seed(config, adapter, interface, obs_dim, total_timesteps, num_seeds):
    train_fn = _build_train_fn(config, total_timesteps)   # Expensive: XLA compile
    results = []
    for seed in range(num_seeds):
        result = _run_single_seed(train_fn, ...)           # Cheap: reuses compiled fn
        results.append(result)
    return _average_results(results)
```

**Metrics returned:**
```python
{
    "success_rate": 0.62,             # Averaged across seeds
    "final_return": 696.5,
    "success_curve": [...],           # Per-eval success during training
    "learning_curve": [...],          # Per-eval reward during training
    "training_time": 377.6,
    "nan_detected": False,
    "per_seed_success": [0.62, 0.41, 0.31],  # Individual seed results
    "seed_spread": 0.31,             # Max - min across seeds
}
```

---

## 12. Discovery Task Wrappers

**Files:** `mdp_discovery/tasks/go1_push_recovery.py`, `mdp_discovery/tasks/panda_pick_and_track.py`

Thin subclasses of the base environment that enrich `state.info` with pre-computed values. This is necessary because the LLM generates standalone functions that cannot call environment methods like `self.get_gyro(data)` — they only have access to `state` and its fields.

### Pattern

```python
class Go1PushRecovery(_BaseGo1PushRecovery):
    def _enrich_info(self, data, info):
        return {
            **info,
            "gyro": self.get_gyro(data),           # (3,) angular velocity
            "gravity": self.get_gravity(data),     # (3,) gravity in body frame
            "local_linvel": self.get_local_linvel(data),  # (3,)
            "upvector": self.get_upvector(data),   # (3,)
            "pos_xy": data.qpos[0:2] - self._origin_xy,  # (2,)
            "height": data.qpos[2],                # ()
            "heading": self._quat_to_yaw(data.qpos[3:7]),  # ()
        }

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(info=self._enrich_info(state.data, state.info))

    def step(self, state, action):
        state = super().step(state, action)
        return state.replace(info=self._enrich_info(state.data, state.info))
```

The enriched `state.info` fields are documented in the context document, which the LLM receives in its prompt. The LLM then generates code like:

```python
def get_observation(state):
    gyro = state.info["gyro"]
    gravity = state.info["gravity"]
    pos_xy = state.info["pos_xy"]
    ...
```

### Currently Implemented Task Wrappers

| Task | Base Class | Enriched `state.info` Fields |
|------|-----------|------------------------------|
| Go1 Push Recovery | `test_go1_pushrecovery.Go1PushRecovery` | gyro, gravity, local_linvel, upvector, pos_xy, height, heading |
| Panda Tracking | `test_panda_tracking.PandaTracking` | target_pos, target_vel, gripper_pos, gripper_target_dist |

---

## 13. Evolution Tracing

**File:** `mdp_discovery/evolution_trace.py`

Logs every evolutionary transition to a JSONL file for post-hoc analysis.

### Event Structure

```python
@dataclass
class EvolutionEvent:
    iteration: int
    timestamp: float
    parent_id: Optional[str]       # None for from-scratch
    child_id: str
    parent_metrics: Optional[Dict]
    child_metrics: Dict
    improvement_delta: Dict        # child - parent for each metric
    obs_dim: int
    island: int
    generation: int
    stage: str                     # "full_train", "short_train", etc.
    model: str                     # LLM model used
    parent_code: Optional[str]     # If include_code=True
    child_code: Optional[str]
    prompt_user: Optional[str]     # If include_prompts=True
```

### Output Format

One JSON object per line in `evolution_trace.jsonl`:

```jsonl
{"iteration": 1, "parent_id": null, "child_id": "abc123", "child_metrics": {"success_rate": 0.12}, ...}
{"iteration": 2, "parent_id": "abc123", "child_id": "def456", "improvement_delta": {"success_rate": 0.33}, ...}
```

Buffered writes (default buffer size 10) for efficiency.

---

## 14. Checkpointing and Resume

### What Gets Saved

Every `checkpoint_interval` iterations (and at the end of the run):

| File | Contents |
|------|----------|
| `controller_state.json` | iteration, token counts, eval time, wall time, task description, config |
| `database/metadata.json` | best_program_id, islands, grid, archive, feature_stats, island_generations |
| `database/programs/*.json` | Individual program entries (code, metrics, obs_dim, feature_coords) |
| `best_interface.py` | Best program's source code with header comment |
| `evolution.log` | Full debug log |
| `evolution_trace.jsonl` | All evolutionary transitions |

### Resume

```bash
python run.py --config configs/task.yaml --resume runs/my_experiment/ --iterations 50
```

The controller:
1. Loads database from `database/` (programs, grid, islands, archive)
2. Restores iteration counter, token counts, eval time from `controller_state.json`
3. Continues from `iteration + 1`

---

## 15. Evolution Modes and Ablations

| Mode | Observation | Reward | Use Case |
|------|-------------|--------|----------|
| `full` | LLM-evolved | LLM-evolved | Main approach: joint optimization |
| `reward_only` | Env default | LLM-evolved | Eureka-equivalent baseline |
| `obs_only` | LLM-evolved | Env built-in | Tests observation contribution |
| `default` | Env default | Env built-in | Lower bound (no evolution, train once) |
| `random` | LLM from-scratch | LLM from-scratch | No parent feedback, tests LLM prior |

**How modes are implemented:**
- `full`: Both functions generated by LLM, both used
- `reward_only`: LLM generates both functions, but evaluator replaces `get_observation` with `adapter.get_default_obs_fn()` before training
- `obs_only`: LLM generates both functions, but evaluator sets `compute_reward = None` (env built-in reward passes through)
- `default`: No LLM call. Single training run with default obs and reward
- `random`: LLM generates from scratch each iteration (no parent code, no top programs, no feedback — only task description and failures)

The prompt builder is mode-aware: in `reward_only` mode, the system prompt omits observation design instructions; in `obs_only` mode, it omits reward design instructions. This focuses the LLM on the relevant component.

---

## 16. Complete Data Flow

```
User provides:
  ├── Task: "Pick up the blue pyramid"
  ├── Context: contexts/xminigrid_context.md
  └── Config: configs/easy_pickup.yaml

Initialization:
  ├── Config loaded from YAML
  ├── Adapter created (XMinigrid or MuJoCo)
  ├── Context document read from file
  ├── Database initialized (empty MAP-Elites grid)
  ├── LLM client created (Bedrock)
  └── Prompt builder created with context + task

Iteration 1 (from scratch):
  ├── No parent → build from-scratch prompt
  ├── LLM generates code
  ├── Crash filter: Stage 0 (AST) → Stage 1 (import) → Stage 2 (dry-run)
  ├── Short training (1M steps) → success = 0.45
  ├── Threshold check: 0.45 > 0.05 ✓
  ├── Full training (5M steps, 3 seeds) → success = 0.62
  └── Stored in database: cell [2,1], island 0

Iteration 2 (evolutionary):
  ├── Sample parent from database → program from iter 1
  ├── Retrieve: top 3, diverse 2, failures 0, inspiration 2
  ├── Build evolutionary prompt with parent code + metrics
  ├── LLM generates improved code
  ├── Crash filter passes
  ├── Short training → success = 0.71
  ├── Full training → success = 0.78
  └── Stored in database: replaces parent in cell [2,1] (better fitness)

... (30 iterations) ...

Iteration 30:
  ├── Best program: 96.8% success, 174-dim obs
  ├── Checkpoint saved
  └── best_interface.py written

Output:
  ├── experiments/main/easy/best_interface.py
  ├── experiments/main/easy/database/ (full population)
  ├── experiments/main/easy/evolution_trace.jsonl
  └── experiments/main/easy/evolution.log
```

---

## 17. File Map

```
mdp_discovery/
├── __init__.py
├── config.py              # Hierarchical dataclass config, YAML loading
├── controller.py          # Main evolution loop, orchestration
├── crash_filter.py        # 3-stage code validation
├── database.py            # MAP-Elites + island model population
├── evaluator.py           # Cascade training pipeline
├── evolution_trace.py     # JSONL event logging
├── llm_client.py          # AWS Bedrock LLM interface
├── mdp_interface.py       # Code loading, validation, wrapping
├── prompts.py             # System/user prompt construction
├── train.py               # PPO for discrete (XLand-MiniGrid)
├── train_brax.py          # PPO for continuous (MuJoCo/Brax)
├── adapters/
│   ├── __init__.py        # get_adapter() factory
│   ├── base.py            # Abstract EnvAdapter
│   └── mujoco_adapter.py  # MuJoCo/Brax adapter + MjxMDPInterfaceWrapper
└── tasks/
    ├── __init__.py
    ├── go1_push_recovery.py    # Go1 enriched state.info
    └── panda_pick_and_track.py # Panda enriched state.info

configs/
├── default.yaml               # XLand-MiniGrid defaults
├── easy_pickup.yaml
├── medium_place_near.yaml
├── hard_rule_chain.yaml
├── go1_push_recovery.yaml
└── panda_pick_and_track.yaml

contexts/
├── xminigrid_context.md       # XLand-MiniGrid API reference
├── go1_push_recovery_context.md
└── panda_pick_and_track_context.md

run.py                         # CLI entry point
test_go1_pushrecovery.py       # Go1 base environment
test_panda_tracking.py         # Panda base environment
profile_go1.py                 # Go1 profiling script
profile_panda.py               # Panda profiling script
```
