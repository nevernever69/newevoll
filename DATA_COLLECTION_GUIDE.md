# Complete Data Collection Guide

## Overview

This guide shows you **everything** that gets saved during a run and how to download it.

## What Gets Saved (With Full Logging Enabled)

### 1. Evolution Trace (`evolution_trace.jsonl`)
**Location**: `experiments/ablation/easy_pickup_no_evolution/evolution_trace.jsonl`

**Contains** (line-separated JSON, one event per line):
```json
{
  "iteration": 1,
  "parent_id": "abc123",
  "child_id": "def456",
  "parent_metrics": {"success_rate": 0.65, "final_length": 45.2},
  "child_metrics": {"success_rate": 0.76, "final_length": 39.1},
  "obs_dim": 17,
  "island": 0,
  "generation": 2,
  "stage": "full_train",
  "model": "claude-4-sonnet",
  "parent_code": "def get_observation(state):\n    ...",  // Full source code
  "child_code": "def get_observation(state):\n    ...",   // Full source code
  "prompt": {
    "system": "You are an expert...",  // Full prompt
    "user": "Given this parent program..."
  },
  "timestamp": "2026-03-03T10:15:30"
}
```

**What you can analyze**:
- Every candidate's full source code
- Parent → child relationships
- Complete prompts sent to LLM
- Metrics for every candidate
- Which candidates improved vs degraded
- Full evolution lineage

**Enable with**:
```yaml
evolution_trace:
  enabled: true
  include_code: true      # ← CRITICAL: Saves full source
  include_prompts: true   # ← CRITICAL: Saves prompts
  buffer_size: 10
```

### 2. Database (`database/`)
**Location**: `experiments/ablation/easy_pickup_no_evolution/database/`

**Contains**:
- `programs.json` - All programs in the archive with full metadata
- `archive.json` - MAP-Elites grid state
- `islands.json` - Island populations

**Example program entry**:
```json
{
  "id": "abc123",
  "code": "def get_observation(state):\n    ...",
  "obs_dim": 17,
  "fitness": 0.76667,
  "metrics": {
    "success_rate": 0.76667,
    "final_length": 79.12917,
    "training_time": 123.45
  },
  "feature_coords": [2, 3],
  "generation": 1,
  "parent_id": null,
  "island": 0,
  "timestamp": "2026-03-03T10:15:30"
}
```

### 3. Controller State (`controller_state.json`)
**Location**: `experiments/ablation/easy_pickup_no_evolution/controller_state.json`

**Contains**:
```json
{
  "iteration": 1,
  "total_llm_tokens": 450000,
  "total_input_tokens": 380000,
  "total_output_tokens": 70000,
  "est_llm_cost_usd": 2.19,
  "total_eval_time": 3245.6,
  "wall_time": 3600.0,
  "task_description": "Pick up the blue pyramid.",
  "evolution_mode": "full",
  "config": { ... }  // Full config used
}
```

### 4. Best Program (`best_interface.py`)
**Location**: `experiments/ablation/easy_pickup_no_evolution/best_interface.py`

**Contains**:
- Best MDP interface found
- Header with metadata (success rate, obs_dim, etc.)
- Full executable Python code

### 5. W&B Artifacts (Automatic)
**Location**: https://wandb.ai/nevernever/mdp-discovery

**Contains**:
- All metrics logged during run
- System metrics (GPU, CPU, memory)
- Best program as artifact
- Training curves
- Summary statistics

## Full Configuration for Maximum Data Collection

Add this to ALL your config files:

```yaml
# Save everything
evolution_trace:
  enabled: true
  include_code: true      # Full source code in traces
  include_prompts: true   # Full prompts in traces
  buffer_size: 10         # Flush every 10 events

# Save more context in prompts
prompt:
  num_top_programs: 3
  num_diverse_programs: 2
  num_failure_examples: 2
  num_inspiration_programs: 5
  include_artifacts: true
  max_artifact_bytes: 20480  # 20KB of training logs per program

# Keep larger archive
database:
  population_size: 200
  archive_size: 50         # Keep top 50 programs
```

## How to Download Everything from EC2

### Option 1: Create Archive and Upload to S3

```bash
# On EC2 - Create comprehensive archive
cd ~/newevoll

# Archive everything
tar -czf results_complete.tar.gz \
  experiments/ablation/*/evolution_trace.jsonl \
  experiments/ablation/*/controller_state.json \
  experiments/ablation/*/best_interface.py \
  experiments/ablation/*/database/

# Check size
ls -lh results_complete.tar.gz

# Upload to S3
aws s3 cp results_complete.tar.gz s3://mdp-discovery-results-nevernever/

# On local machine - Download
aws s3 cp s3://mdp-discovery-results-nevernever/results_complete.tar.gz .
tar -xzf results_complete.tar.gz
```

### Option 2: Sync Entire Experiments Directory

```bash
# On EC2 - Upload everything
aws s3 sync experiments/ablation/ s3://mdp-discovery-results-nevernever/ablation/ \
  --exclude "*.pyc" \
  --exclude "__pycache__/*"

# On local machine - Download everything
aws s3 sync s3://mdp-discovery-results-nevernever/ablation/ ./ec2_results/ablation/
```

### Option 3: Download Specific Files

```bash
# Evolution trace (most important - has everything!)
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/easy_pickup_no_evolution/evolution_trace.jsonl .

# Database
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/easy_pickup_no_evolution/database/ ./database/ --recursive

# Best program
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/easy_pickup_no_evolution/best_interface.py .

# Controller state
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/easy_pickup_no_evolution/controller_state.json .
```

## File Sizes (Estimates)

With `include_code: true` and `include_prompts: true`:

| File | Without Code/Prompts | With Code/Prompts |
|------|---------------------|-------------------|
| `evolution_trace.jsonl` | ~100 KB | **~5-10 MB** |
| `database/programs.json` | ~200 KB | ~500 KB |
| `controller_state.json` | ~10 KB | ~50 KB |
| `best_interface.py` | ~5 KB | ~5 KB |
| **Total per run** | ~315 KB | **~6-11 MB** |

For 5 tasks: ~30-55 MB total (very manageable!)

## Analyzing Evolution Trace

The `evolution_trace.jsonl` file is the **gold mine**. It has everything!

### Python Script to Extract All Candidates

```python
import json

# Load all events
events = []
with open('evolution_trace.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Extract all unique candidates
candidates = {}
for event in events:
    child_id = event['child_id']
    candidates[child_id] = {
        'code': event['child_code'],
        'metrics': event['child_metrics'],
        'obs_dim': event['obs_dim'],
        'generation': event['generation'],
        'parent_id': event['parent_id'],
    }

print(f"Total candidates: {len(candidates)}")

# Find best candidate
best_id = max(candidates, key=lambda x: candidates[x]['metrics']['success_rate'])
best = candidates[best_id]

print(f"\nBest candidate (ID: {best_id}):")
print(f"Success rate: {best['metrics']['success_rate']:.1%}")
print(f"Obs dim: {best['obs_dim']}")
print(f"\nCode:\n{best['code']}")

# Save all candidates to individual files
import os
os.makedirs('all_candidates', exist_ok=True)
for cid, data in candidates.items():
    with open(f'all_candidates/candidate_{cid}.py', 'w') as f:
        f.write(f"# Success rate: {data['metrics']['success_rate']:.1%}\n")
        f.write(f"# Obs dim: {data['obs_dim']}\n")
        f.write(f"# Generation: {data['generation']}\n\n")
        f.write(data['code'])

print(f"\nSaved {len(candidates)} candidates to all_candidates/")
```

### Extract All Prompts

```python
import json

with open('evolution_trace.jsonl', 'r') as f:
    events = [json.loads(line) for line in f]

# Save first prompt (from-scratch)
first_event = events[0]
if 'prompt' in first_event:
    with open('prompts/prompt_from_scratch.txt', 'w') as f:
        f.write("=== SYSTEM ===\n")
        f.write(first_event['prompt']['system'])
        f.write("\n\n=== USER ===\n")
        f.write(first_event['prompt']['user'])

# Save evolutionary prompt (with parent)
if len(events) > 10:
    evo_event = events[10]
    if 'prompt' in evo_event:
        with open('prompts/prompt_with_parent.txt', 'w') as f:
            f.write("=== SYSTEM ===\n")
            f.write(evo_event['prompt']['system'])
            f.write("\n\n=== USER ===\n")
            f.write(evo_event['prompt']['user'])

print("Prompts saved to prompts/")
```

## Download Checklist

After your run completes:

- [ ] `evolution_trace.jsonl` - **Most important! Has everything!**
- [ ] `database/programs.json` - All programs in archive
- [ ] `controller_state.json` - Run summary and costs
- [ ] `best_interface.py` - Best program found
- [ ] W&B metrics (already online at wandb.ai)

## Pro Tips

1. **Always enable `include_code` and `include_prompts`** - disk is cheap, data is precious
2. **Use S3 for automatic backup** - add this to your run script:
   ```bash
   # After run completes
   aws s3 sync experiments/ s3://mdp-discovery-results-nevernever/experiments/
   ```
3. **Parse `evolution_trace.jsonl` locally** - easier than SSH'ing to EC2 repeatedly
4. **Keep raw files** - compress them (`tar -czf`) but don't delete
5. **Check file sizes** before downloading - `du -sh experiments/ablation/*/`

## Quick Download Script

Save this as `download_results.sh` on your **local machine**:

```bash
#!/bin/bash
# Download all results from S3

RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
    echo "Usage: ./download_results.sh easy_pickup_no_evolution"
    exit 1
fi

mkdir -p results/$RUN_NAME

echo "Downloading results for: $RUN_NAME"

# Evolution trace (most important!)
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/$RUN_NAME/evolution_trace.jsonl \
  results/$RUN_NAME/

# Database
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/$RUN_NAME/database/ \
  results/$RUN_NAME/database/ --recursive

# Summary files
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/$RUN_NAME/controller_state.json \
  results/$RUN_NAME/
aws s3 cp s3://mdp-discovery-results-nevernever/ablation/$RUN_NAME/best_interface.py \
  results/$RUN_NAME/

echo "✓ Downloaded to results/$RUN_NAME/"
ls -lh results/$RUN_NAME/
```

Usage:
```bash
chmod +x download_results.sh
./download_results.sh easy_pickup_no_evolution
./download_results.sh easy_pickup_full_evolution
```
