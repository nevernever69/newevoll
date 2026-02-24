#!/bin/bash
# =============================================================================
# Ablation Experiments: obs_only and reward_only for Easy, Medium, Hard
#
# Same setup as main experiments:
#   30 iterations × 2 candidates × 3 training seeds per candidate
#
# For each task, runs:
#   1. obs_only   — evolve observation only, use env's built-in reward
#   2. reward_only — evolve reward only, use default flat observation
#
# Order: easy (obs, rew) → medium (obs, rew) → hard (obs, rew)
# =============================================================================
set -e

EXPERIMENT_DIR="experiments/ablations"
ITERATIONS=30
CANDIDATES=2
NUM_SEEDS=3
MODEL="claude-sonnet-4-6"

mkdir -p "$EXPERIMENT_DIR/easy_obs_only" \
        "$EXPERIMENT_DIR/easy_reward_only" \
        "$EXPERIMENT_DIR/medium_obs_only" \
        "$EXPERIMENT_DIR/medium_reward_only" \
        "$EXPERIMENT_DIR/hard_obs_only" \
        "$EXPERIMENT_DIR/hard_reward_only"

TOTAL_START=$(date +%s)

echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "Ablation Experiments — $(date)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Iterations: $ITERATIONS" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Candidates: $CANDIDATES per iteration" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Training seeds: $NUM_SEEDS per candidate" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Model: $MODEL" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Modes: obs_only, reward_only" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# EASY — obs_only (SKIPPED — already complete)
# ─────────────────────────────────────────────────────────────────
EASY_OBS_TIME=0
echo "[$(date +%H:%M:%S)] EASY obs_only: SKIPPED (already complete)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# EASY — reward_only
# ─────────────────────────────────────────────────────────────────
EASY_REW_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> EASY reward_only: Pick up blue pyramid (default obs, evolve reward)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] easy/reward_only starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/easy_pickup.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --mode reward_only \
    --task "Pick up the blue pyramid. Grid 9x9, max_steps=80." \
    --run-dir "$EXPERIMENT_DIR/easy_reward_only" \
    2>&1 | tee "$EXPERIMENT_DIR/easy_reward_only/stdout.log"

EASY_REW_END=$(date +%s)
EASY_REW_TIME=$((EASY_REW_END - EASY_REW_START))
echo "[$(date +%H:%M:%S)] EASY reward_only complete: ${EASY_REW_TIME}s ($(echo "scale=1; $EASY_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# MEDIUM — obs_only
# ─────────────────────────────────────────────────────────────────
MED_OBS_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> MEDIUM obs_only: Place pyramid near square (evolve obs, env reward)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] medium/obs_only starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/medium_place_near.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --mode obs_only \
    --task "Place the yellow pyramid adjacent to the green square. Grid 9x9, max_steps=80." \
    --run-dir "$EXPERIMENT_DIR/medium_obs_only" \
    2>&1 | tee "$EXPERIMENT_DIR/medium_obs_only/stdout.log"

MED_OBS_END=$(date +%s)
MED_OBS_TIME=$((MED_OBS_END - MED_OBS_START))
echo "[$(date +%H:%M:%S)] MEDIUM obs_only complete: ${MED_OBS_TIME}s ($(echo "scale=1; $MED_OBS_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# MEDIUM — reward_only
# ─────────────────────────────────────────────────────────────────
MED_REW_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> MEDIUM reward_only: Place pyramid near square (default obs, evolve reward)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] medium/reward_only starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/medium_place_near.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --mode reward_only \
    --task "Place the yellow pyramid adjacent to the green square. Grid 9x9, max_steps=80." \
    --run-dir "$EXPERIMENT_DIR/medium_reward_only" \
    2>&1 | tee "$EXPERIMENT_DIR/medium_reward_only/stdout.log"

MED_REW_END=$(date +%s)
MED_REW_TIME=$((MED_REW_END - MED_REW_START))
echo "[$(date +%H:%M:%S)] MEDIUM reward_only complete: ${MED_REW_TIME}s ($(echo "scale=1; $MED_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# HARD — obs_only
# ─────────────────────────────────────────────────────────────────
HARD_OBS_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> HARD obs_only: Rule chain + placement (evolve obs, env reward)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] hard/obs_only starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/hard_rule_chain.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --mode obs_only \
    --task "Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400." \
    --run-dir "$EXPERIMENT_DIR/hard_obs_only" \
    2>&1 | tee "$EXPERIMENT_DIR/hard_obs_only/stdout.log"

HARD_OBS_END=$(date +%s)
HARD_OBS_TIME=$((HARD_OBS_END - HARD_OBS_START))
echo "[$(date +%H:%M:%S)] HARD obs_only complete: ${HARD_OBS_TIME}s ($(echo "scale=1; $HARD_OBS_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# HARD — reward_only
# ─────────────────────────────────────────────────────────────────
HARD_REW_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> HARD reward_only: Rule chain + placement (default obs, evolve reward)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] hard/reward_only starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/hard_rule_chain.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --mode reward_only \
    --task "Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400." \
    --run-dir "$EXPERIMENT_DIR/hard_reward_only" \
    2>&1 | tee "$EXPERIMENT_DIR/hard_reward_only/stdout.log"

HARD_REW_END=$(date +%s)
HARD_REW_TIME=$((HARD_REW_END - HARD_REW_START))
echo "[$(date +%H:%M:%S)] HARD reward_only complete: ${HARD_REW_TIME}s ($(echo "scale=1; $HARD_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "ABLATION EXPERIMENTS COMPLETE — $(date)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Easy   obs_only:    ${EASY_OBS_TIME}s ($(echo "scale=1; $EASY_OBS_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Easy   reward_only: ${EASY_REW_TIME}s ($(echo "scale=1; $EASY_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Medium obs_only:    ${MED_OBS_TIME}s ($(echo "scale=1; $MED_OBS_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Medium reward_only: ${MED_REW_TIME}s ($(echo "scale=1; $MED_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Hard   obs_only:    ${HARD_OBS_TIME}s ($(echo "scale=1; $HARD_OBS_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Hard   reward_only: ${HARD_REW_TIME}s ($(echo "scale=1; $HARD_REW_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Total:              ${TOTAL_TIME}s ($(echo "scale=1; $TOTAL_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"

# Aggregate cost
python3 -c "
import json, glob

total_input = 0
total_output = 0
total_cost = 0.0
print()
print('--- Per-run cost breakdown ---')
for f in sorted(glob.glob('$EXPERIMENT_DIR/*/controller_state.json')):
    with open(f) as fh:
        meta = json.load(fh)
    task = f.split('/')[-2]
    inp = meta.get('total_input_tokens', 0)
    out = meta.get('total_output_tokens', 0)
    cost = meta.get('est_llm_cost_usd', 0)
    wall = meta.get('wall_time', 0)
    total_input += inp
    total_output += out
    total_cost += cost
    print(f'  {task:25s}: input={inp:>10,} output={out:>8,} cost=\${cost:.2f} wall={wall:.0f}s')

print()
print(f'  TOTAL input:  {total_input:,}')
print(f'  TOTAL output: {total_output:,}')
print(f'  TOTAL cost:   \${total_cost:.2f}')

with open('$EXPERIMENT_DIR/summary.json', 'w') as f:
    json.dump({
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_est_cost_usd': round(total_cost, 2),
        'total_wall_time_s': $TOTAL_TIME,
    }, f, indent=2)
print(f'  Saved to $EXPERIMENT_DIR/summary.json')
" | tee -a "$EXPERIMENT_DIR/experiment.log"
