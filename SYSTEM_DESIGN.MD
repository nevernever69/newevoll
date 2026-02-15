# PureEvol: MDP Interface Discovery via LLM-Guided Evolutionary Search

## 1. Research Question

Can the MDP interface — the observation mapping, action mapping, and reward function — be treated as a learnable object? Can LLM-guided meta-optimization automatically construct effective task formulations that induce desired behavior in environments where no reward function is provided, given only a natural language task description and access to the environment's library/API?

### Sub-Questions

1. Does jointly discovering the full interface (obs + reward) outperform discovering reward alone (Eureka-style)?
2. What structural properties do discovered interfaces exhibit compared to human-designed ones?
3. Which interface components (observation vs. reward) contribute most to task performance?
4. Does evolutionary search outperform random LLM sampling?

---

## 2. Core Design Principle

**The context document is the single, complete source of all environment-specific knowledge.**

The system code (prompts, crash filter, training loop, evaluator) contains **zero** environment-specific hints, constants, or patterns. All env knowledge — state fields, tile constants, action spaces, API patterns, code examples — lives in the user-provided context doc, which is a "condensed codebase" of the target environment framework.

This means:
- The system prompts are generic RL engineering prompts with no env-specific vocabulary
- The template defines only the interface contract (function signatures + type constraints + JAX rules)
- The context doc is injected verbatim as a "Library Reference" in the system prompt
- To support a new environment, you write a context doc + an adapter — no system code changes

### User Inputs

| Input | What it contains | Example |
|-------|-----------------|---------|
| **Task description** | Natural language goal | "Navigate to the goal tile in a 6x6 empty room." |
| **Context document** | Condensed env framework reference — state fields, constants, action space, API patterns, code examples | `contexts/xminigrid_context.md` |
| **Config file** | Training hyperparameters, env ID, adapter type, evolution mode | `configs/default.yaml` |

### System Components (env-agnostic)

| Component | What it does | Env-specific content |
|-----------|-------------|---------------------|
| `prompts.py` | Builds LLM prompts | None — generic RL engineering advice |
| `crash_filter.py` | 3-stage code validation | None — takes `dummy_state` as parameter |
| `train.py` | PPO training loop | None — takes wrapped env from adapter |
| `evaluator.py` | Cascade evaluation pipeline | None — delegates to adapter |
| `controller.py` | Main evolution loop | None — creates adapter from config |
| `templates/` | Interface contract | None — function signatures + JAX rules only |

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
                        USER INPUTS
   ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐
   │ Task          │  │ Context Doc      │  │ Config        │
   │ Description   │  │ (condensed       │  │ (YAML)        │
   │ (natural lang)│  │  codebase)       │  │               │
   └──────┬───────┘  └────────┬─────────┘  └───────┬───────┘
          │                   │                     │
          ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUTION CONTROLLER                       │
│                                                               │
│  1. Creates Environment Adapter (from config.adapter_type)   │
│  2. Loads context doc (from config.context_file)             │
│  3. Initializes PromptBuilder, Evaluator, Database           │
│  4. Runs evolution loop or baseline mode                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐    ┌──────────────┐   ┌──────────────┐
   │   LLM    │    │  Cascade     │   │  MAP-Elites  │
   │Generation│───▶│  Evaluator   │──▶│  Database     │
   │ Engine   │    │              │   │  + Islands    │
   └──────────┘    └──────────────┘   └──────┬───────┘
                                              │
                              feedback loop ──┘
```

### 3.2 Detailed Flow Per Iteration

```
1. Sample parent from database (or generate from scratch)
       │
2. Build prompt:
   - System: generic RL prompt + context doc as Library Reference
   - User: task + parent code + training feedback + top/diverse programs + failures
       │
3. Call LLM (AWS Bedrock Claude) → extract code from ```python``` fence
       │
4. Crash Filter (3 stages):
   - Stage 0: ast.parse + verify required functions exist
   - Stage 1: importlib load (catches bad imports)
   - Stage 2: dry-run with adapter.get_dummy_state() (catches shape/type errors)
       │  FAIL → store error artifact, feed back to LLM
       │
5. Cascade Training:
   - Short training (1M steps) → threshold check (5% success)
   - If passed → Full training (5M steps)
   - Final dedicated success evaluation (50 episodes)
       │
6. Store in MAP-Elites database
   - Feature dimensions: reward_complexity × obs_dimensionality
   - Island assignment + migration management
       │
7. Checkpoint (every N iterations)
```

---

## 4. Repository Structure

```
pureevol/
├── run.py                              # CLI entry point
├── configs/
│   └── default.yaml                    # Default config (MiniGrid-Empty-6x6)
├── contexts/
│   └── xminigrid_context.md            # XLand-MiniGrid library reference
├── templates/
│   └── mdp_interface_template.py       # Interface contract (signatures + constraints)
├── mdp_discovery/
│   ├── __init__.py
│   ├── config.py                       # Hierarchical dataclass config + YAML loading
│   ├── controller.py                   # Main evolution loop + baseline modes
│   ├── evaluator.py                    # Cascade evaluator pipeline
│   ├── crash_filter.py                 # 3-stage code validation
│   ├── train.py                        # PPO training (JAX pmap)
│   ├── train_utils.py                  # Rollout, GAE, PPO update functions
│   ├── nn.py                           # ActorCriticMLP with GRU
│   ├── mdp_interface.py                # Interface loader + validator
│   ├── prompts.py                      # Prompt templates + PromptBuilder
│   ├── llm_client.py                   # AWS Bedrock Claude client
│   ├── database.py                     # MAP-Elites + island evolution
│   └── adapters/
│       ├── __init__.py                 # Factory: get_adapter(type, config)
│       ├── base.py                     # EnvAdapter ABC
│       └── xminigrid_adapter.py        # XMinigrid implementation + MDPInterfaceWrapper
└── SYSTEM_DESIGN.MD                    # This document
```

### Key Isolation Property

```bash
# Only ONE file imports xminigrid:
$ grep -r "import xminigrid" mdp_discovery/
mdp_discovery/adapters/xminigrid_adapter.py

# Core modules have ZERO env-specific imports:
$ grep -r "import xminigrid" mdp_discovery/*.py
(no results)
```

---

## 5. Component Details

### 5.1 Environment Adapter (`mdp_discovery/adapters/`)

The adapter is the **only** place environment-specific code lives. All other modules interact with the environment through this interface.

**Abstract interface** (`base.py`):

```python
class EnvAdapter(ABC):
    def make_env(get_obs_fn, reward_fn) -> (env, env_params)
    def make_eval_env(get_obs_fn, reward_fn, max_steps) -> (env, env_params)
    def get_dummy_state() -> state
    def compute_success(rollout_stats, env_params) -> jax.Array
    def get_default_obs_fn() -> Callable
    def get_default_reward_fn() -> Optional[Callable]  # None = env built-in
    def num_actions(env_params) -> int
```

**XMinigrid implementation** (`xminigrid_adapter.py`):
- Contains `MDPInterfaceWrapper(Wrapper)` — replaces obs and/or reward in the env step
- Wrapper supports partial wrapping: if `get_obs_fn is None`, passes through default obs; if `reward_fn is None`, keeps env reward
- `make_env`: creates base env → MDPInterfaceWrapper → GymAutoResetWrapper
- `compute_success`: `(episode_length < max_steps)` — xminigrid terminates early on goal
- `get_default_obs_fn`: flattened grid (normalized) + position + one-hot direction + pocket + step count

**Factory** (`__init__.py`):
```python
def get_adapter(adapter_type: str, config) -> EnvAdapter:
    if adapter_type == "xminigrid":
        return XMinigridAdapter(config)
    raise ValueError(...)
```

### 5.2 Configuration (`config.py`)

Hierarchical dataclass config with YAML loading via `dacite`:

```python
@dataclass
class Config:
    max_iterations: int = 500
    candidates_per_iteration: int = 1
    evolution_mode: str = "full"          # full | reward_only | obs_only | default | random
    task_description: Optional[str] = None

    environment: EnvironmentConfig        # env_id, adapter_type, context_file
    training: TrainingConfig              # PPO hyperparameters
    mdp_interface: MDPInterfaceConfig     # max_obs_dim
    crash_filter: CrashFilterConfig       # stage timeouts
    evaluator: EvaluatorConfig            # cascade thresholds
    llm: LLMConfig                        # model, region, temperature
    database: DatabaseConfig              # MAP-Elites params, island config
    prompt: PromptConfig                  # num_top/diverse/failure programs
```

Key config fields for env-agnosticism:
- `environment.adapter_type`: which adapter to use (e.g., `"xminigrid"`)
- `environment.context_file`: path to context doc (e.g., `"contexts/xminigrid_context.md"`)
- `evolution_mode`: controls what the LLM generates and what defaults are used

### 5.3 Prompts (`prompts.py`)

**System prompt** — assembled at init, env-agnostic:
```
You are an expert RL engineer. You design MDP interfaces for RL agents
in reinforcement learning environments.

{function_description}           ← mode-dependent (full: both functions, reward_only: just reward, etc.)

[Design philosophy — generic RL advice about obs/reward design]

## Library Reference
{library_context}                ← context doc injected verbatim

## Output Format
{output_constraints}             ← mode-dependent constraints
```

**User prompt — from scratch**:
```
## Task
{task_description}

## Instructions
{function_instructions}          ← mode-dependent
{obs_instructions}               ← omitted in reward_only mode
{reward_instructions}            ← omitted in obs_only mode
[JAX pitfalls]
[Constraints]
```

**User prompt — evolutionary**:
```
## Task
## Parent Program (code + metrics)
## Training Feedback (success analysis, plateau detection, curve trends)
## Best Known Programs
## Diverse Approaches
## Recent Failures (crashes + low-fitness)
## Improvement Guidance (mode-aware, success-level-dependent)
```

All guidance is generic: "progress-based shaping", "task-relevant state features", "refer to the library reference". No env-specific vocabulary (no "grid", "Manhattan", "tiles", etc.).

### 5.4 MDP Interface (`mdp_interface.py`)

Loads LLM-generated code via tempfile + importlib:

```python
@dataclass
class MDPInterface:
    get_observation: Optional[Callable]   # None if not required by mode
    compute_reward: Optional[Callable]    # None if not required by mode
    obs_dim: Optional[int]
    source_code: Optional[str]
```

- `from_code(code, required_functions=["get_observation", "compute_reward"])` — load from string
- `validate(dummy_state, max_obs_dim)` — dry-run, check shapes/types, detect obs_dim
- Supports partial loading: `required_functions=["compute_reward"]` in reward_only mode

### 5.5 Crash Filter (`crash_filter.py`)

Three stages, fully parameterized (no env imports):

| Stage | What it checks | Input needed |
|-------|---------------|-------------|
| 0 | `ast.parse` + required function names exist | `code`, `required_functions` |
| 1 | `importlib` load — catches bad imports, JAX errors | `code`, `required_functions` |
| 2 | Dry-run with real state — catches shape/attribute/tracing errors | `interface`, `dummy_state` |

```python
def run_crash_filter(code, config, dummy_state, required_functions=None) -> CrashFilterResult
```

### 5.6 Training (`train.py`)

PPO with GRU, adapted from xland-minigrid's reference implementation:

- `make_states(config, adapter, interface, obs_dim, total_timesteps)` — builds env via `adapter.make_env()`, creates network + optimizer
- `make_train(env, env_params, config, total_timesteps, compute_success_fn)` — returns JIT-compiled pmap'd training function
- `run_training(config, adapter, interface, obs_dim, total_timesteps)` — full pipeline: make_states → replicate → train → final eval → return metrics
- `_run_success_eval(config, adapter, interface, obs_dim, train_state)` — dedicated final evaluation using `adapter.make_eval_env()` and `adapter.compute_success()`

Key: success is computed via `adapter.compute_success()`, not hardcoded. The `compute_success_fn` is passed into the pmap'd function so it works inside JIT.

### 5.7 Cascade Evaluator (`evaluator.py`)

Pipeline: crash filter → short training → threshold → full training.

```python
class CascadeEvaluator:
    def __init__(self, config, adapter: EnvAdapter):
        self.adapter = adapter
        self.dummy_state = adapter.get_dummy_state()

    def evaluate(self, code: str) -> CandidateResult:
        required_functions = self._get_required_functions()  # mode-dependent
        # Stage 1: crash filter (with dummy_state from adapter)
        # Mode overrides: reward_only → inject adapter.get_default_obs_fn()
        #                 obs_only → inject adapter.get_default_reward_fn()
        # Stage 2+3: short train → threshold → full train
```

### 5.8 Controller (`controller.py`)

Main evolution loop with 3 dispatch modes:

```python
class EvolutionController:
    def run(self):
        if mode == "default":   → self._run_default_baseline()   # no LLM, adapter defaults
        if mode == "random":    → self._run_random_baseline()    # LLM from scratch each time, no evolution
        else:                   → standard evolution loop         # full, reward_only, obs_only
```

- Creates adapter via `get_adapter(config.environment.adapter_type, config)`
- Loads context from `config.environment.context_file`
- Passes `evolution_mode` to `PromptBuilder` for mode-aware prompt generation
- Handles checkpointing, crash retries, island rotation, migration

### 5.9 Template (`templates/mdp_interface_template.py`)

Pure interface contract — no env-specific content:

```python
"""
Constraints:
  - get_observation must return 1D jnp.float32 (max 512 elements)
  - compute_reward must return scalar jnp.float32
  - JAX-traceable only
  - Only jax/jax.numpy imports
"""
def get_observation(state) -> jnp.ndarray: ...
def compute_reward(state, action, next_state) -> jnp.ndarray: ...
```

### 5.10 CLI (`run.py`)

```bash
# Basic run
python run.py --task "Navigate to the goal tile." --iterations 50

# Ablation modes
python run.py --task "..." --mode reward_only
python run.py --task "..." --mode obs_only
python run.py --task "..." --mode default --iterations 1
python run.py --task "..." --mode random --iterations 10

# Overrides
python run.py --task "..." --env MiniGrid-Empty-8x8 --context contexts/custom.md
python run.py --task "..." --config configs/doorkey.yaml --model claude-4-sonnet
python run.py --task "..." --no-cascade --timesteps-full 10000000

# Resume from checkpoint
python run.py --task "..." --resume runs/previous_run/
```

---

## 6. Evolution Modes

| Mode | LLM generates | Crash filter checks | Obs fn used | Reward fn used | Purpose |
|------|--------------|-------------------|-------------|---------------|---------|
| `full` | obs + reward | both | LLM-generated | LLM-generated | Main experiment |
| `reward_only` | reward only | compute_reward only | adapter default | LLM-generated | Eureka-style baseline |
| `obs_only` | obs only | get_observation only | LLM-generated | env built-in | Obs contribution ablation |
| `default` | nothing | nothing | adapter default | env built-in | No-design-effort baseline |
| `random` | obs + reward | both | LLM-generated | LLM-generated | No-evolution baseline |

### Mode Behavior in Each Component

**PromptBuilder**: `_FUNCTION_DESC` and `_OUTPUT_CONSTRAINTS` dicts select mode-specific instructions. Obs design instructions omitted in `reward_only`. Reward design instructions omitted in `obs_only`.

**CascadeEvaluator.evaluate()**: After loading the interface:
- `reward_only` → `interface.get_observation = adapter.get_default_obs_fn()`
- `obs_only` → `interface.compute_reward = adapter.get_default_reward_fn()` (None → wrapper keeps env reward)

**Controller.run()**: `default` → train once with adapter defaults, no LLM. `random` → generate from scratch every iteration, no parent/feedback.

---

## 7. Evolutionary Database

### MAP-Elites Grid

Feature dimensions:
- **reward_complexity**: Character count of `compute_reward` function body (proxy for structural complexity)
- **obs_dimensionality**: Size of observation vector returned by `get_observation`

Each cell stores the best program with those feature coordinates. This maintains diversity — simple 6-feature observations coexist with complex 50-feature ones.

### Island Model

- 3 islands, each with its own MAP-Elites grid
- Islands evolve independently for `migration_interval` generations
- Migration: top programs from each island copy to neighbors (ring topology)
- Prevents premature convergence on one observation/reward strategy

### Per-Program Storage

- Full source code
- Fitness (success rate)
- Training metrics (learning curves, episode lengths, training time)
- Feature coordinates
- Parent ID + generation number
- Island ID + timestamp

---

## 8. Context Document

The context document (`contexts/xminigrid_context.md`) is a comprehensive, manually written reference covering:

1. **State Object** — every field with types, shapes, and descriptions
2. **Tile Constants** — all 13 tile types with IDs and categories
3. **Color Constants** — all 12 colors with IDs
4. **Action Space** — 6 discrete actions with effects
5. **Direction Vectors** — direction IDs, delta vectors, front-position calculation
6. **Goal Encoding** — 15 goal types with encoding format and conditions
7. **Rule Encoding** — 12 rule types with encoding format and effects
8. **Step Execution Order** — action → rules → step_num → goal check → truncation → reward → obs
9. **Coordinate System** — (row, col) indexing, y-down convention
10. **JAX Constraints** — do's and don'ts with code examples
11. **Available Imports** — only `jax` and `jax.numpy`
12. **Useful Patterns** — distance calculations, tile counting, position finding, normalization
13. **Registered Environments** — env IDs, grid sizes, descriptions

The context doc contains **no reward design hints** and **no observation design suggestions**. It is purely an API reference.

---

## 9. Baselines and Ablations

### 9.1 Baselines

| Baseline | Method | What it tests |
|----------|--------|--------------|
| **Default** (`--mode default`) | No LLM. Adapter's default obs + env built-in reward | "No design effort" lower bound |
| **Random** (`--mode random`) | LLM generates from scratch each iteration, no evolution | Does evolutionary feedback matter? |
| **Human-designed** | Hand-written obs + reward by RL engineer | Standard practice upper bound |

### 9.2 Ablation Modes

| Ablation | Mode flag | What it isolates |
|----------|-----------|-----------------|
| **Full system** | `--mode full` | Joint obs + reward discovery |
| **Reward only** | `--mode reward_only` | Reward discovery with fixed obs (Eureka-style) |
| **Obs only** | `--mode obs_only` | Obs discovery with env default reward |

### 9.3 Experimental Protocol

For each environment × mode × seed:
```
experiments/
  main/{env}/seed_{s}/              # full mode
  ablation/{mode}/{env}/seed_{s}/   # reward_only, obs_only
  baseline/{mode}/{env}/seed_{s}/   # default, random
```

---

## 10. Environments

### 10.1 Current: XLand-MiniGrid

All environments use the xminigrid adapter. JAX-compiled, stateless, vectorized.

| Environment | Grid | Description | Difficulty |
|-------------|------|-------------|------------|
| MiniGrid-Empty-6x6 | 6x6 | Navigate to goal in empty room | Easy |
| MiniGrid-Empty-5x5 | 5x5 | Smaller empty room | Easy |
| MiniGrid-Empty-8x8 | 8x8 | Larger empty room | Easy-Medium |
| MiniGrid-DoorKey-5x5 | 5x5 | Pick up key, unlock door, reach goal | Medium |
| MiniGrid-DoorKey-6x6 | 6x6 | Larger door-key puzzle | Medium-Hard |
| MiniGrid-FourRooms | 19x19 | Navigate through connected rooms | Hard |
| MiniGrid-Unlock | 11x11 | Find key and unlock door | Medium-Hard |

### 10.2 Adding New Environments

To support a new environment framework:

1. **Write a context doc** — comprehensive API reference for the framework
2. **Write an adapter** — implement `EnvAdapter` ABC in `mdp_discovery/adapters/`
3. **Register in factory** — add to `get_adapter()` in `adapters/__init__.py`
4. **Create config** — YAML with `adapter_type`, `context_file`, training hyperparams

No changes needed to prompts, crash filter, training loop, evaluator, or controller.

---

## 11. Validated Results

### Test Run: Empty-6x6 (3 iterations, post-refactor)

All 3 iterations achieved **100% success rate** with optimal episode length 7.0.

| Iter | Gen | obs_dim | Obs Strategy | Reward Strategy | Success | Converge |
|------|-----|---------|-------------|----------------|---------|----------|
| 1 | 0 | 11 | pos + one-hot dir + goal pos + manhattan + euclidean + step | euclidean shaping ×1.0 + penalty -0.01 + bonus ×10 | 100% | Update 3/38 |
| 2 | 1 | 9 | pos + one-hot dir + relative goal + manhattan | manhattan shaping ×0.1 + bonus ×1.0 | 100% | Update 4/38 |
| 3 | 1 | 7 | pos + sin/cos dir + goal direction + euclidean | euclidean shaping ×0.5 + bonus ×2.0 | 100% | Update 3/38 |

**Key observations:**
- The LLM correctly uses the context doc (meshgrid goal-finding, proper casting, JAX-compatible code)
- Evolution produces meaningful diversity: Iter 2 and 3 are structurally different from Iter 1
- MAP-Elites fills different cells: (2,2) for Iter 1, (0,0) for Iter 2 and 3
- Empty-6x6 is saturated — harder environments needed for meaningful comparison

---

## 12. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Context doc delivery | Manual, user-provided | Reproducible, controlled. No agentic exploration variance. |
| Env-specific isolation | Adapter pattern | Single file per env. Core system is 100% env-agnostic. |
| Interface format | Two fixed-signature functions | Constrains structure enough to run, flexible enough for discovery. |
| Template content | Pure contract (no env content) | Context doc is the sole source of env knowledge. |
| Inner-loop algorithm | PPO with GRU | Standard, well-understood, fast in JAX. Not the variable under study. |
| Evolutionary method | MAP-Elites + islands | Maintains diversity of approaches. Prevents premature convergence. |
| Cascade evaluation | 1M short / 5M full | Filters bad candidates cheaply. 5% threshold empirically validated. |
| Success metric | Task completion rate | Clear, unambiguous, not gameable by reward hacking. |
| LLM backend | AWS Bedrock Claude | Strong code generation, handles JAX well. |
| Evolution modes | 5 modes (full/reward_only/obs_only/default/random) | Covers all ablations needed for paper experiments. |

---

## 13. Comparison with Eureka

| Aspect | PureEvol | Eureka |
|--------|----------|--------|
| What's evolved | Observation + Reward (joint) | Reward only |
| Starting point | From scratch (no seed) | Environment source code as context |
| Env framework | XLand-MiniGrid (JAX) | IsaacGym (PyTorch) |
| Training backend | PPO with GRU (JAX pmap) | rl_games PPO (PyTorch) |
| Environments | 7 grid-world tasks | 35 dexterous manipulation tasks |
| Diversity mechanism | MAP-Elites + islands | Top-K sampling |
| Env-agnostic | Yes (adapter pattern) | Partially (prompts are generic, training is IsaacGym-locked) |
| Context source | Manual context doc | Raw env source code passed to LLM |
| Action transform | Not yet (planned) | Not supported |

**Key differentiator**: PureEvol discovers the **full MDP interface** (obs + reward), not just reward. The hypothesis is that joint optimization discovers synergies between observation design and reward shaping that reward-only methods cannot.

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Joint discovery shows no benefit over reward-only | Weakens main contribution | Valid negative result. Paper pivots to analysis of when joint helps vs. hurts. |
| Empty-6x6 too easy (all methods hit 100%) | No differentiation between approaches | Test on harder envs (DoorKey, FourRooms, Unlock). |
| LLM generates memorized patterns | Undermines "discovery" claim | Analyze code diversity. Custom envs reduce memorization risk. |
| Evolution doesn't beat random sampling | Evolution component unnecessary | Report honestly. Contribution becomes "LLM-based MDP generation." |
| Adapter abstraction has performance overhead | Slower training | Adapter just creates envs at init — no per-step overhead. Validated: ~25s per 5M steps. |
