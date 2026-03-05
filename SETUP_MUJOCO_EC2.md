# MuJoCo Playground Setup for EC2

## Problem
Go1 training was failing because mujoco_playground wasn't set up correctly. The task files expect a local clone in the repo root.

## Solution
The official mujoco_playground (installed as "playground" package) auto-downloads menagerie assets on first use. Clone it locally:

```bash
# On EC2, in ~/newevoll directory
cd ~/newevoll

# Clone mujoco_playground into repo root
git clone https://github.com/google-deepmind/mujoco_playground.git

# Verify the directory structure
ls -la mujoco_playground/

# The task files (mdp_discovery/tasks/go1_push_recovery.py) automatically add
# this directory to sys.path, so no pip install needed

# Assets will auto-download on first environment load
```

## How It Works
1. `mdp_discovery/tasks/go1_push_recovery.py` line 15-17 adds `mujoco_playground/` to sys.path
2. Imports work directly from the cloned repo
3. When you first run Go1, menagerie assets download automatically
4. No separate mujoco_menagerie installation needed

## Test the Setup
```bash
cd ~/newevoll
source .venv/bin/activate

# Test with cascade disabled (full ablation)
python run.py --config configs/go1_push_recovery_noevo.yaml

# Or test with diagnostic config (only 3 candidates)
python run.py --config configs/go1_test_cascade.yaml
```

## Expected Behavior
- First run: Assets download automatically (one-time delay)
- Training: Each candidate should take 5-6 minutes for 30M steps
- Success metrics: Should see various % success rates, not all 0%

## If Still Failing
Check logs for actual error messages (not just empty "Full training failed:")
```bash
# Run with DEBUG logging
python run.py --config configs/go1_test_cascade.yaml
```
