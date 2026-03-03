# Ablation Study Analysis Guide

## Overview

This guide shows how to compare **No Evolution** vs **Full Evolution** results using Weights & Biases.

## Experimental Setup

### No Evolution (Random Sampling)
- **Config**: `configs/easy_pickup_noevo.yaml`
- **Strategy**: 1 iteration × 30 candidates
- **Mechanism**: Generate 30 independent MDP interfaces with no feedback
- **Hypothesis**: Random sampling might be sufficient for simple tasks

### Full Evolution (Evolutionary Search)
- **Config**: `configs/easy_pickup_fullevo.yaml`
- **Strategy**: 10 iterations × 3 candidates = 30 total
- **Mechanism**: Iteratively improve using:
  - Parent program feedback
  - Elite programs from archive
  - Diversity-based inspiration
  - Failure examples
- **Hypothesis**: Evolution should find better solutions than random

### Fair Comparison
Both approaches use **exactly 30 candidate evaluations** (same computational budget).

## Running the Comparison

### On EC2 Instance

```bash
# Option 1: Run in screen (recommended for long experiments)
screen -S ablation_comparison
cd ~/newevoll
source .venv/bin/activate
export WANDB_ENABLED=1
bash run_ablation_comparison.sh

# Detach: Ctrl+A, then D
# Reattach: screen -r ablation_comparison

# Option 2: Run directly (only if you'll stay connected)
cd ~/newevoll
source .venv/bin/activate
export WANDB_ENABLED=1
bash run_ablation_comparison.sh
```

### Expected Runtime (on g5.2xlarge)
- **No Evolution**: ~1-2 hours (if not already complete)
- **Full Evolution**: ~2-3 hours
- **Total**: ~3-5 hours

## Analyzing Results in W&B

### 1. Access Your Project
Go to: https://wandb.ai/nevernever/mdp-discovery

### 2. Compare Runs
You should see two runs:
- `easy_pickup_no_evolution` (tag: `ablation-no-evolution`)
- `easy_pickup_full_evolution` (tag: `mode-full`)

Click the checkbox next to both runs, then click **"Compare"** button.

### 3. Key Metrics to Compare

#### Performance Metrics (Primary)
| Metric | Description | Better |
|--------|-------------|--------|
| `best/success_rate` | Final best success rate achieved | Higher ↑ |
| `best/obs_dim` | Observation dimension of best solution | Lower ↓ (simpler) |
| `best/episode_length` | Episode length of best solution | Task-dependent |

#### Efficiency Metrics (Secondary)
| Metric | Description | Better |
|--------|-------------|--------|
| `final/wall_time` | Total wall clock time | Lower ↓ |
| `final/total_llm_tokens` | Total tokens used (cost proxy) | Lower ↓ |
| `final/est_llm_cost_usd` | Estimated LLM cost | Lower ↓ |

#### Process Metrics (Insight)
| Metric | Description | Insight |
|--------|-------------|---------|
| `candidate_progress` | Success rate over candidates | Shows improvement trajectory |
| `iteration/candidates_passed` | Candidates per iteration that passed | Shows quality of generation |

### 4. Create Comparison Charts

#### Chart 1: Success Rate Comparison (Bar Chart)
- X-axis: Run name (no_evolution, full_evolution)
- Y-axis: `best/success_rate`
- **Interpretation**: Which approach found better solutions?

#### Chart 2: Progress Over Time (Line Chart)
- X-axis: Candidate number (0-30)
- Y-axis: `best/success_rate`
- Lines: One per run
- **Interpretation**: Does evolution show steady improvement?

#### Chart 3: Cost vs Performance (Scatter)
- X-axis: `final/est_llm_cost_usd`
- Y-axis: `best/success_rate`
- Points: One per run
- **Interpretation**: Cost-effectiveness comparison

### 5. Download Summary Table

In W&B, go to **Table** view and export with these columns:
```
Run Name | best/success_rate | best/obs_dim | final/wall_time | final/total_llm_tokens | final/est_llm_cost_usd
```

### 6. Example Analysis Questions

#### Q1: Does evolution help?
**Method**: Compare `best/success_rate`
- If Full Evolution > No Evolution by 5%+: **Evolution helps**
- If difference < 5%: **Inconclusive** (random is competitive)
- If No Evolution > Full Evolution: **Random better** (surprising!)

#### Q2: Is evolution worth the complexity?
**Method**: Consider both performance and cost
- If Full Evolution has higher success BUT also higher cost/time: **Trade-off**
- If Full Evolution dominates on both: **Clear winner**
- If similar performance but No Evolution is cheaper: **Random preferred**

#### Q3: How does improvement happen?
**Method**: Plot `candidate_progress` over time
- No Evolution: Should be flat or random walk (no improvement)
- Full Evolution: Should show upward trend (learning from parents)

## Expected Results

### Scenario A: Evolution Works
```
No Evolution:    76% success, $X cost, Y time
Full Evolution:  88% success, $X cost, Y time
Conclusion: Evolution provides 12% improvement
```

### Scenario B: Random is Sufficient
```
No Evolution:    76% success, $X cost, Y time
Full Evolution:  78% success, $X cost, 2×Y time
Conclusion: Random sampling is competitive and faster
```

### Scenario C: Task-Dependent
```
Easy tasks:   Random ≈ Evolution (simple enough)
Medium tasks: Evolution > Random (needs refinement)
Hard tasks:   Evolution >> Random (critical for success)
```

## Creating a W&B Report

### Steps:
1. In your W&B project, click **"Create Report"**
2. Add sections:
   - **Introduction**: Describe ablation study
   - **Metrics Comparison**: Add bar charts for key metrics
   - **Progress Over Time**: Add line charts showing improvement
   - **System Metrics**: GPU usage, memory (optional)
   - **Conclusion**: Summarize findings
3. Click **"Share"** to get a public link
4. Save the report (File → Export PDF)

### Report Template Structure:
```markdown
# Ablation Study: Evolution vs Random Sampling

## Hypothesis
Does evolutionary feedback improve MDP interface discovery over random sampling?

## Setup
- Task: easy_pickup (pick up blue pyramid)
- Budget: 30 candidates each
- No Evolution: 1 iter × 30 candidates
- Full Evolution: 10 iters × 3 candidates

## Results
[Insert comparison charts]

### Performance
- No Evolution: X% success
- Full Evolution: Y% success
- Delta: (Y-X)%

### Efficiency
- Time: A vs B seconds
- Cost: $C vs $D

## Conclusion
[Your analysis here]
```

## Troubleshooting

### Issue: Runs not showing up in W&B
**Solution**: Check `WANDB_ENABLED=1` was set before running

### Issue: Can't compare runs with different number of iterations
**Solution**: Use `candidate` or `step` as X-axis instead of `iteration`

### Issue: Metrics missing
**Solution**: Check `controller_state.json` in run directories for raw data

## Next Steps

After analyzing easy_pickup:
1. **Extend to other tasks**: Run same comparison for medium/hard tasks
2. **Statistical significance**: Run multiple seeds (3-5 runs each)
3. **Vary budget**: Test with 10, 20, 50 candidates
4. **Hybrid approaches**: Combine random + evolution
