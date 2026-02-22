#!/bin/bash
# =============================================================================
# Final Main Experiment: Hard only (Easy + Medium already done)
# 30 iterations × 2 candidates × 3 training seeds per candidate
# =============================================================================
set -e

EXPERIMENT_DIR="experiments/main"
ITERATIONS=30
CANDIDATES=2
NUM_SEEDS=3
MODEL="claude-sonnet-4-6"

mkdir -p "$EXPERIMENT_DIR"

TOTAL_START=$(date +%s)

echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "Continuing Experiment (Hard only) — $(date)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Iterations: $ITERATIONS" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Candidates: $CANDIDATES per iteration" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Training seeds: $NUM_SEEDS per candidate" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Model: $MODEL" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Skipping: easy, medium (already complete)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# EASY + MEDIUM — SKIPPED (already complete)
# ─────────────────────────────────────────────────────────────────
EASY_TIME=0
MED_TIME=0

# ─────────────────────────────────────────────────────────────────
# HARD
# ─────────────────────────────────────────────────────────────────
HARD_START=$(date +%s)
echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo ">>> HARD: Rule chain + TileNearGoal (13x13, 4 rooms)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "[$(date +%H:%M:%S)] hard starting" | tee -a "$EXPERIMENT_DIR/experiment.log"

python3 run.py \
    --config configs/hard_rule_chain.yaml \
    --iterations $ITERATIONS \
    --candidates $CANDIDATES \
    --num-seeds $NUM_SEEDS \
    --model "$MODEL" \
    --task "Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400." \
    --run-dir "$EXPERIMENT_DIR/hard" \
    2>&1 | tee "$EXPERIMENT_DIR/hard/stdout.log"

HARD_END=$(date +%s)
HARD_TIME=$((HARD_END - HARD_START))
echo "[$(date +%H:%M:%S)] HARD complete: ${HARD_TIME}s ($(echo "scale=1; $HARD_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "EXPERIMENT COMPLETE — $(date)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Easy:   ${EASY_TIME}s ($(echo "scale=1; $EASY_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Medium: ${MED_TIME}s ($(echo "scale=1; $MED_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Hard:   ${HARD_TIME}s ($(echo "scale=1; $HARD_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "  Total:  ${TOTAL_TIME}s ($(echo "scale=1; $TOTAL_TIME/60" | bc)m)" | tee -a "$EXPERIMENT_DIR/experiment.log"
echo "============================================================" | tee -a "$EXPERIMENT_DIR/experiment.log"

# Aggregate cost from controller_state.json files
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
    parts = f.split('/')
    task = parts[-2]
    inp = meta.get('total_input_tokens', 0)
    out = meta.get('total_output_tokens', 0)
    cost = meta.get('est_llm_cost_usd', 0)
    wall = meta.get('wall_time', 0)
    total_input += inp
    total_output += out
    total_cost += cost
    print(f'  {task}: input={inp:,} output={out:,} cost=\${cost:.2f} wall={wall:.0f}s')

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
