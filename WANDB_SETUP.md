# Weights & Biases Integration

W&B integration for real-time tracking of ablation experiments.

## Features

- **Real-time metrics**: Success rates, episode lengths, obs_dim
- **Candidate tracking**: Monitor all 30 candidates in parallel
- **Best program tracking**: Automatic logging of improvements
- **Batch summaries**: Pass/crash rates, average eval time
- **Code artifacts**: Saves best_interface.py for each run
- **Cost tracking**: LLM token usage and cost estimates
- **Auto-tagging**: Ablation runs are automatically tagged

## Quick Setup (EC2)

### 1. Install W&B

```bash
cd ~/newevoll
source .venv/bin/activate
pip install wandb
```

### 2. Login to W&B

```bash
wandb login
```

This will prompt you for your W&B API key. Get it from: https://wandb.ai/authorize

### 3. Enable W&B Logging

```bash
export WANDB_ENABLED=1
```

Add to your `~/.bashrc` to make it persistent:
```bash
echo 'export WANDB_ENABLED=1' >> ~/.bashrc
```

### 4. Run Your Ablation

```bash
source .venv/bin/activate
export WANDB_ENABLED=1

python run.py \
  --config configs/easy_pickup_noevo.yaml \
  --run-dir experiments/ablation/easy_pickup_no_evolution
```

That's it! Your run will now log to W&B.

## What Gets Logged

### Real-Time Metrics (per candidate)

- `success_rate`: Training success rate
- `episode_length`: Average episode length
- `obs_dim`: Observation dimension
- `stage`: Evaluation stage (short_train, full_train, rejected, crashed)
- `training_time`: Time spent training
- `candidate_progress`: Progress through batch (0.0 to 1.0)
- `new_best`: Flag when new best is found

### Best Program Tracking

- `best/success_rate`: Success rate of current best
- `best/obs_dim`: Observation dimension of best
- `best/episode_length`: Episode length of best

### Batch Summaries (per iteration)

- `summary/candidates_generated`: Total candidates in batch
- `summary/candidates_passed`: Number that passed crash filter
- `summary/candidates_crashed`: Number that crashed
- `summary/pass_rate`: Pass rate (passed / generated)
- `summary/best_success_rate`: Best success rate so far
- `summary/avg_eval_time`: Average evaluation time per candidate
- `summary/total_llm_tokens`: Cumulative LLM tokens used

### Final Summary

- `final/wall_time`: Total wall clock time
- `final/total_programs`: Total programs evaluated
- `final/total_eval_time`: Total evaluation time
- `final/total_llm_tokens`: Total LLM tokens
- `final/input_tokens`: Input tokens
- `final/output_tokens`: Output tokens
- `final/est_llm_cost_usd`: Estimated LLM cost in USD
- `final/best_success_rate`: Final best success rate
- `final/best_obs_dim`: Final best obs_dim

### Artifacts

- `best_interface.py`: The winning code artifact

## Viewing Results

### W&B Dashboard

Go to: https://wandb.ai/your-username/mdp-discovery

You'll see:
- **Charts**: Real-time success rates, candidate progress
- **Runs comparison**: Compare all ablation tasks
- **Artifacts**: Download best_interface.py files
- **System metrics**: GPU utilization, memory usage

### Key Charts to Create

1. **Success Rate Over Candidates**
   - X-axis: `candidate`
   - Y-axis: `success_rate`
   - Shows how each of the 30 candidates performs

2. **Best Success Rate Over Time**
   - X-axis: `candidate`
   - Y-axis: `best/success_rate`
   - Shows improvement as better candidates are found

3. **Pass vs Crash Rate**
   - Pie chart: `summary/candidates_passed` vs `summary/candidates_crashed`

4. **Obs Dim Distribution**
   - Histogram: `obs_dim`
   - Shows variety in observation dimensions

## Disabling W&B

```bash
unset WANDB_ENABLED
# or
export WANDB_ENABLED=0
```

The system will run normally without logging to W&B.

## Tags

Runs are automatically tagged:
- `ablation-no-evolution`: For configs with "noevo" in path
- `mode-{mode}`: Evolution mode (full, reward_only, obs_only)

## Project Organization

```
W&B Project: mdp-discovery
│
├── Run 1: easy_pickup_no_evolution
│   ├── Metrics (30 candidates)
│   ├── Best tracking
│   └── Artifact: best_interface.py
│
├── Run 2: medium_place_near_no_evolution
│   └── ...
│
└── Run 3: hard_rule_chain_no_evolution
    └── ...
```

## Troubleshooting

### "wandb not installed"

```bash
pip install wandb
```

### "Not logged in"

```bash
wandb login
```

### W&B not logging (but no errors)

Check that `WANDB_ENABLED=1`:
```bash
echo $WANDB_ENABLED
```

### Want to run without W&B

```bash
unset WANDB_ENABLED
```

Or just don't install wandb - the code will gracefully skip logging.

## Comparing Ablation Runs

In W&B dashboard:

1. Select all your ablation runs (easy, medium, hard, go1, panda)
2. Click "Compare runs"
3. Add charts:
   - `final/best_success_rate` (bar chart)
   - `summary/pass_rate` (line chart)
   - `final/total_llm_tokens` (bar chart)

This lets you see which tasks benefited most from evolution vs no-evolution.

## Cost Tracking

The `final/est_llm_cost_usd` metric shows estimated AWS Bedrock costs based on:
- Claude Sonnet pricing: $3/M input tokens, $15/M output tokens

Use W&B's grouping to see total costs across all ablation runs.

## Advanced: Custom Metrics

You can add custom W&B logging in your code:

```python
import wandb

# Log custom metric
wandb.log({"my_metric": value})

# Log image
wandb.log({"visualization": wandb.Image(image_array)})
```

## Support

- W&B Docs: https://docs.wandb.ai/
- W&B Community: https://community.wandb.ai/
- Issues with integration: Check `evolution.log` for warnings

---

**Ready?** `pip install wandb` → `wandb login` → `export WANDB_ENABLED=1` → Run!
