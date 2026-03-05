#!/bin/bash
# Run medium and hard tasks, then shutdown instance
set -e

echo "========================================"
echo "Running Medium + Hard Tasks"
echo "========================================"
echo "Task 1: medium_place_near (30 candidates)"
echo "Task 2: hard_rule_chain (30 candidates)"
echo "Then: Auto-shutdown instance"
echo "========================================"
echo ""

# Activate environment
source .venv/bin/activate
export WANDB_ENABLED=1

# Track start time
TOTAL_START=$(date +%s)

# Task 1: Medium
echo "========================================"
echo "Starting: medium_place_near"
echo "========================================"
TASK1_START=$(date +%s)

python run.py \
  --config configs/medium_place_near_noevo.yaml \
  --run-dir experiments/ablation/medium_place_near_no_evolution

TASK1_END=$(date +%s)
TASK1_TIME=$((TASK1_END - TASK1_START))
echo ""
echo "✓ medium_place_near complete in ${TASK1_TIME}s (~$((TASK1_TIME / 60)) min)"
echo ""

# Task 2: Hard
echo "========================================"
echo "Starting: hard_rule_chain"
echo "========================================"
TASK2_START=$(date +%s)

python run.py \
  --config configs/hard_rule_chain_noevo.yaml \
  --run-dir experiments/ablation/hard_rule_chain_no_evolution

TASK2_END=$(date +%s)
TASK2_TIME=$((TASK2_END - TASK2_START))
echo ""
echo "✓ hard_rule_chain complete in ${TASK2_TIME}s (~$((TASK2_TIME / 60)) min)"
echo ""

# Summary
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "========================================"
echo "ALL TASKS COMPLETE"
echo "========================================"
echo "medium_place_near: ${TASK1_TIME}s (~$((TASK1_TIME / 60)) min)"
echo "hard_rule_chain:   ${TASK2_TIME}s (~$((TASK2_TIME / 60)) min)"
echo "Total time:        ${TOTAL_TIME}s (~$((TOTAL_TIME / 60)) min)"
echo ""
echo "Results saved in:"
echo "  experiments/ablation/medium_place_near_no_evolution/"
echo "  experiments/ablation/hard_rule_chain_no_evolution/"
echo ""
echo "View on W&B: https://wandb.ai/nevernever/mdp-discovery"
echo ""
echo "========================================"
echo "Uploading results to S3..."
echo "========================================"

# Create archive
tar -czf medium_hard_results.tar.gz \
  experiments/ablation/medium_place_near_no_evolution/ \
  experiments/ablation/hard_rule_chain_no_evolution/

# Upload to S3 (optional - remove if you don't want this)
# aws s3 cp medium_hard_results.tar.gz s3://mdp-discovery-results-nevernever/

echo "✓ Results archived: medium_hard_results.tar.gz"
echo ""
echo "========================================"
echo "SHUTTING DOWN INSTANCE IN 10 SECONDS"
echo "========================================"
echo "Press Ctrl+C to cancel shutdown"
sleep 10

# Shutdown instance
echo "Shutting down now..."
sudo shutdown -h now
