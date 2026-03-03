#!/bin/bash
# Run No Evolution Ablation on ALL 5 Tasks
# Each task: 30 independent candidates, all logged to W&B
# Total: 150 candidates for comprehensive analysis

set -e

# Activate virtual environment
source .venv/bin/activate

# Enable W&B logging
export WANDB_ENABLED=1

echo "========================================"
echo "NO EVOLUTION ABLATION - ALL 5 TASKS"
echo "========================================"
echo "Strategy: 30 independent candidates per task"
echo "No evolution, no parent feedback"
echo "All candidates logged to W&B"
echo "Total candidates: 150 (30 × 5 tasks)"
echo "========================================"
echo ""

# Track timing
TOTAL_START=$(date +%s)

# Define tasks
TASKS=(
    "easy_pickup:easy_pickup_noevo"
    "medium_place_near:medium_place_near_noevo"
    "hard_rule_chain:hard_rule_chain_noevo"
    "go1_push_recovery:go1_push_recovery_noevo"
    "panda_pick_and_track:panda_pick_and_track_noevo"
)

COMPLETED=0
FAILED=0

for TASK_INFO in "${TASKS[@]}"; do
    IFS=':' read -r TASK_NAME CONFIG_NAME <<< "$TASK_INFO"

    echo "========================================"
    echo "Task: ${TASK_NAME}"
    echo "Config: configs/${CONFIG_NAME}.yaml"
    echo "========================================"

    START=$(date +%s)

    # Check if already completed
    if [ -d "experiments/ablation/${TASK_NAME}_no_evolution" ]; then
        echo "⚠️  Run already exists: experiments/ablation/${TASK_NAME}_no_evolution"
        read -p "Skip this task? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping ${TASK_NAME}..."
            ((COMPLETED++))
            continue
        fi
    fi

    # Run the task
    echo "Starting ${TASK_NAME}..."
    if python run.py \
        --config configs/${CONFIG_NAME}.yaml \
        --run-dir experiments/ablation/${TASK_NAME}_no_evolution; then

        END=$(date +%s)
        DURATION=$((END - START))
        echo ""
        echo "✓ ${TASK_NAME} complete in ${DURATION}s"
        ((COMPLETED++))

        # Extract result
        if [ -f "experiments/ablation/${TASK_NAME}_no_evolution/controller_state.json" ]; then
            echo "Results saved to: experiments/ablation/${TASK_NAME}_no_evolution/"
        fi
    else
        echo ""
        echo "✗ ${TASK_NAME} FAILED"
        ((FAILED++))
    fi

    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "========================================"
echo "ABLATION STUDY COMPLETE"
echo "========================================"
echo "Completed: ${COMPLETED}/5 tasks"
echo "Failed: ${FAILED}/5 tasks"
echo "Total time: ${TOTAL_DURATION}s (~$((TOTAL_DURATION / 60)) minutes)"
echo ""
echo "Results saved in: experiments/ablation/"
echo ""
echo "Next steps:"
echo "1. Upload to S3:"
echo "   aws s3 sync experiments/ablation/ s3://mdp-discovery-results-nevernever/ablation/"
echo ""
echo "2. View on W&B:"
echo "   https://wandb.ai/nevernever/mdp-discovery"
echo ""
echo "3. Analyze all 150 candidates:"
echo "   - Compare success rate distributions across tasks"
echo "   - Find best candidate per task"
echo "   - Visualize obs_dim vs success_rate"
echo "   - Check which tasks are harder (lower median success)"
echo ""
echo "4. Download results:"
echo "   aws s3 sync s3://mdp-discovery-results-nevernever/ablation/ ./results/"
echo "========================================"
