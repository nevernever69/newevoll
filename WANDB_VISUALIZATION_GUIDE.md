# W&B Visualization Guide for 150 Candidates

## Overview

After running all 5 tasks with 30 candidates each, you'll have **150 independent candidates** logged to W&B.

This guide shows you how to visualize and analyze all of them.

## What Gets Logged to W&B

### Per Candidate (150 total entries)
Each of the 150 candidates logs:
- `success_rate` - Main performance metric
- `obs_dim` - Observation dimension
- `episode_length` - Average episode length
- `training_time` - Time to train
- `stage` - full_train, short_train, rejected, or crashed
- `candidate` - Candidate number (1-30)
- `iteration` - Always 1 (no evolution)
- `is_best` - Whether this is the best for this task

### Per Run (5 runs)
Each task run logs:
- `best/success_rate` - Best candidate found
- `best/obs_dim` - Obs dim of best candidate
- `best/episode_length` - Episode length of best
- `final/total_llm_tokens` - Total tokens used
- `final/est_llm_cost_usd` - Estimated cost
- `final/wall_time` - Total time taken

## Accessing W&B

Go to: https://wandb.ai/nevernever/mdp-discovery

You should see 5 runs:
- `easy_pickup_no_evolution`
- `medium_place_near_no_evolution`
- `hard_rule_chain_no_evolution`
- `go1_push_recovery_no_evolution`
- `panda_pick_and_track_no_evolution`

All tagged with: `ablation-no-evolution`

## Key Visualizations to Create

### 1. Success Rate Distribution by Task

**Chart Type**: Box Plot or Violin Plot

**Steps**:
1. Click "Create Report" or "Visualizations"
2. Add new panel → Custom Chart
3. X-axis: `run.name` (or group by run)
4. Y-axis: `success_rate`
5. Chart type: Box plot or Violin

**What to look for**:
- Which task has highest median success rate? (easier)
- Which task has most variance? (more randomness)
- Which task has lowest max success rate? (harder)
- Are there outliers?

**Example interpretation**:
```
easy_pickup:        Median 70%, Range 60-80%  → Easy, consistent
medium_place_near:  Median 45%, Range 20-70%  → Moderate, variable
hard_rule_chain:    Median 10%, Range 0-25%   → Hard, difficult to solve
```

### 2. All 150 Candidates Scatter Plot

**Chart Type**: Scatter Plot

**Steps**:
1. Add new panel → Custom Chart
2. X-axis: `candidate` (1-30)
3. Y-axis: `success_rate`
4. Color by: `run.name`
5. Size by: `obs_dim` (optional)

**What to look for**:
- Clustering by task (easy tasks cluster high, hard tasks cluster low)
- Spread within each task (variance of sampling)
- Best candidate per task

**Interpretation**:
- Horizontal spread within a color = variance of independent sampling
- No trend left-to-right = truly independent (not improving)

### 3. Observation Dimension vs Success Rate

**Chart Type**: Scatter Plot

**Steps**:
1. X-axis: `obs_dim`
2. Y-axis: `success_rate`
3. Color by: `run.name`

**What to look for**:
- Is there a correlation? (higher obs_dim → better success?)
- Do different tasks cluster at different obs_dims?
- Optimal obs_dim range per task

**Expected patterns**:
- Too small obs_dim (< 10): May lack information
- Too large obs_dim (> 50): May be overfitting or redundant
- Sweet spot: Task-dependent, often 15-30

### 4. Success Rate Histogram per Task

**Chart Type**: Histogram (5 separate panels)

**Steps**:
1. Create 5 separate histograms, one per task
2. X-axis: `success_rate` (bins of 0.1)
3. Y-axis: Count of candidates

**What to look for**:
- Distribution shape (normal, bimodal, skewed?)
- Where does the peak occur?
- Outliers on high/low end

**Example interpretation**:
```
easy_pickup: Normal distribution around 70% → consistent
hard_rule_chain: Right-skewed (most near 0%) → very hard
```

### 5. Cost vs Performance

**Chart Type**: Scatter Plot (run-level, not candidate-level)

**Steps**:
1. X-axis: `final/est_llm_cost_usd`
2. Y-axis: `best/success_rate`
3. Label: `run.name`

**What to look for**:
- Cost-effectiveness per task
- Which task has best ROI (high success / low cost)

### 6. Training Time Distribution

**Chart Type**: Box Plot

**Steps**:
1. X-axis: `run.name`
2. Y-axis: `training_time`
3. Chart type: Box plot

**What to look for**:
- Which task takes longest to train?
- Variance in training time (some candidates timeout?)

### 7. Stage Distribution (Crashed vs Passed)

**Chart Type**: Stacked Bar Chart

**Steps**:
1. X-axis: `run.name`
2. Y-axis: Count
3. Color by: `stage` (full_train, rejected, crashed)

**What to look for**:
- Crash rate per task
- How many candidates reach full training?
- Which task has most rejections?

**Healthy pattern**: Most candidates full_train, few crashed

### 8. Best Candidate per Task (Table)

**Chart Type**: Table

**Columns**:
- Task name
- Best success rate
- Best obs_dim
- Best episode length
- Candidate number (which of 30 was best)
- Cost
- Time

**How to create**:
1. In W&B, select all 5 runs
2. Go to "Table" view
3. Add columns: `best/success_rate`, `best/obs_dim`, etc.
4. Export as CSV or screenshot

## Creating a Comprehensive Report

### Report Structure

```markdown
# No Evolution Ablation: 150 Independent Candidates Across 5 Tasks

## Hypothesis
Can independent sampling (30 random attempts) solve MDP interface discovery
without evolutionary feedback?

## Methodology
- 5 tasks: easy_pickup, medium_place_near, hard_rule_chain, go1, panda
- 30 independent candidates per task
- Same from-scratch prompt for all candidates within a task
- Total budget: 150 candidates

## Results

### Overall Performance
[Insert: Success rate box plot by task]

Key findings:
- easy_pickup: Median X%, Range Y-Z%
- medium_place_near: ...
- hard_rule_chain: ...

### Distribution Analysis
[Insert: Histogram for each task]

Observations:
- Easy tasks show normal distribution (consistent performance)
- Hard tasks show right-skewed distribution (mostly failures)

### Observation Space Analysis
[Insert: obs_dim vs success_rate scatter]

Findings:
- Optimal obs_dim for easy_pickup: 15-20
- Optimal obs_dim for hard_rule_chain: 25-35

### Best Candidates
[Insert: Table of best per task]

### Cost-Benefit Analysis
[Insert: Cost vs performance scatter]

## Conclusions

1. **Task Difficulty Ranking**:
   - Easy: ...
   - Moderate: ...
   - Hard: ...

2. **Independent Sampling Effectiveness**:
   - Works well for easy tasks (>70% median)
   - Struggles with hard tasks (<20% median)
   - High variance suggests more samples could help

3. **Recommendations**:
   - For easy tasks: 10-15 samples sufficient
   - For hard tasks: Need evolution or >50 samples
```

## Python Analysis Script

Download your W&B data and analyze locally:

```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize W&B API
api = wandb.Api()
runs = api.runs("nevernever/mdp-discovery", filters={"tags": "ablation-no-evolution"})

# Collect all candidate data
all_candidates = []
for run in runs:
    task_name = run.name.replace("_no_evolution", "")
    history = run.scan_history(keys=["success_rate", "obs_dim", "episode_length", "candidate", "stage"])

    for row in history:
        if "success_rate" in row:
            all_candidates.append({
                "task": task_name,
                "candidate_num": row.get("candidate", 0),
                "success_rate": row["success_rate"],
                "obs_dim": row.get("obs_dim", 0),
                "episode_length": row.get("episode_length", 0),
                "stage": row.get("stage", "unknown"),
            })

df = pd.DataFrame(all_candidates)

# Save to CSV
df.to_csv("all_150_candidates.csv", index=False)
print(f"Saved {len(df)} candidates to all_150_candidates.csv")

# Summary statistics
print("\nSummary by Task:")
print(df.groupby("task")["success_rate"].describe())

# Best per task
print("\nBest Candidate per Task:")
best_per_task = df.loc[df.groupby("task")["success_rate"].idxmax()]
print(best_per_task[["task", "candidate_num", "success_rate", "obs_dim"]])

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Box plot
sns.boxplot(data=df, x="task", y="success_rate", ax=axes[0, 0])
axes[0, 0].set_title("Success Rate Distribution by Task")
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Scatter: all candidates
for task in df["task"].unique():
    task_df = df[df["task"] == task]
    axes[0, 1].scatter(task_df["candidate_num"], task_df["success_rate"],
                       label=task, alpha=0.6)
axes[0, 1].set_xlabel("Candidate Number")
axes[0, 1].set_ylabel("Success Rate")
axes[0, 1].set_title("All 150 Candidates")
axes[0, 1].legend()

# 3. obs_dim vs success_rate
for task in df["task"].unique():
    task_df = df[df["task"] == task]
    axes[0, 2].scatter(task_df["obs_dim"], task_df["success_rate"],
                       label=task, alpha=0.6)
axes[0, 2].set_xlabel("Observation Dimension")
axes[0, 2].set_ylabel("Success Rate")
axes[0, 2].set_title("Obs Dim vs Success Rate")
axes[0, 2].legend()

# 4-5. Histograms for first two tasks
for i, task in enumerate(df["task"].unique()[:2]):
    task_df = df[df["task"] == task]
    axes[1, i].hist(task_df["success_rate"], bins=20, edgecolor='black')
    axes[1, i].set_xlabel("Success Rate")
    axes[1, i].set_ylabel("Count")
    axes[1, i].set_title(f"{task} Distribution")

# 6. Stage distribution
stage_counts = df.groupby(["task", "stage"]).size().unstack(fill_value=0)
stage_counts.plot(kind="bar", stacked=True, ax=axes[1, 2])
axes[1, 2].set_title("Stage Distribution by Task")
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("ablation_analysis.png", dpi=300)
print("\nPlots saved to ablation_analysis.png")
```

## Quick Checks

### Check 1: Are all 150 candidates logged?
```python
import wandb
api = wandb.Api()
runs = api.runs("nevernever/mdp-discovery", filters={"tags": "ablation-no-evolution"})

for run in runs:
    history = list(run.scan_history(keys=["candidate"]))
    print(f"{run.name}: {len(history)} candidates logged")
# Expected: 30 for each run
```

### Check 2: What's the best candidate overall?
```python
best_success = 0
best_task = None
best_candidate = None

for run in runs:
    best = run.summary.get("best/success_rate", 0)
    if best > best_success:
        best_success = best
        best_task = run.name

print(f"Best: {best_task} with {best_success:.1%}")
```

## Next Steps After Visualization

1. **Identify patterns**: Which tasks benefit from more samples?
2. **Compare to evolution**: Run same analysis on evolution runs
3. **Optimize sampling**: Should we run more candidates for hard tasks?
4. **Publication**: Export W&B report as PDF for paper
