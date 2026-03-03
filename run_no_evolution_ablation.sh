#!/bin/bash
# Run "No Evolution" Ablation Study
# This script runs all 5 tasks with max_iterations=1 and candidates_per_iteration=30
# Usage: bash run_no_evolution_ablation.sh

set -e  # Exit on error

echo "========================================="
echo "No Evolution Ablation Study"
echo "========================================="
echo "Running 5 tasks with 30 independent LLM candidates each"
echo "No iterative evolution - just picking the best from 30 samples"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Activating..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "ERROR: Virtual environment not found. Run setup_ec2.sh first."
        exit 1
    fi
fi

# Check AWS credentials
if [[ -z "$AWS_ACCESS_KEY_ID" ]] && [[ ! -f ~/.aws/credentials ]]; then
    echo "WARNING: AWS credentials not found. Please configure:"
    echo "  aws configure"
    echo "Or set environment variables:"
    echo "  export AWS_ACCESS_KEY_ID='your-key'"
    echo "  export AWS_SECRET_ACCESS_KEY='your-secret'"
    read -p "Press Enter to continue or Ctrl+C to abort..."
fi

# Create experiments directory
mkdir -p experiments/ablation

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="experiments/ablation/no_evolution_ablation_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Function to run a task
run_task() {
    local config=$1
    local run_dir=$2
    local task_name=$3

    echo "========================================="
    echo "Task: $task_name"
    echo "Config: $config"
    echo "Output: $run_dir"
    echo "Started: $(date)"
    echo "========================================="

    python run.py \
        --config "$config" \
        --run-dir "$run_dir" \
        2>&1 | tee -a "$LOG_FILE"

    local status=$?
    echo ""
    echo "Task $task_name completed with status: $status"
    echo "Finished: $(date)"
    echo ""

    return $status
}

# Track overall success
FAILED_TASKS=()

# 1. Easy Pickup (XMinigrid)
echo -e "\n[1/5] Running Easy Pickup..."
if ! run_task \
    "configs/easy_pickup_noevo.yaml" \
    "experiments/ablation/easy_pickup_no_evolution" \
    "Easy Pickup"; then
    FAILED_TASKS+=("Easy Pickup")
fi

# 2. Medium Place Near (XMinigrid)
echo -e "\n[2/5] Running Medium Place Near..."
if ! run_task \
    "configs/medium_place_near_noevo.yaml" \
    "experiments/ablation/medium_place_near_no_evolution" \
    "Medium Place Near"; then
    FAILED_TASKS+=("Medium Place Near")
fi

# 3. Hard Rule Chain (XMinigrid)
echo -e "\n[3/5] Running Hard Rule Chain..."
if ! run_task \
    "configs/hard_rule_chain_noevo.yaml" \
    "experiments/ablation/hard_rule_chain_no_evolution" \
    "Hard Rule Chain"; then
    FAILED_TASKS+=("Hard Rule Chain")
fi

# 4. Go1 Push Recovery (MuJoCo)
echo -e "\n[4/5] Running Go1 Push Recovery (GPU-intensive)..."
if ! run_task \
    "configs/go1_push_recovery_noevo.yaml" \
    "experiments/ablation/go1_no_evolution" \
    "Go1 Push Recovery"; then
    FAILED_TASKS+=("Go1 Push Recovery")
fi

# 5. Panda Pick and Track (MuJoCo)
echo -e "\n[5/5] Running Panda Pick and Track (GPU-intensive)..."
if ! run_task \
    "configs/panda_pick_and_track_noevo.yaml" \
    "experiments/ablation/panda_no_evolution" \
    "Panda Pick and Track"; then
    FAILED_TASKS+=("Panda Pick and Track")
fi

# Summary
echo "========================================="
echo "Ablation Study Complete!"
echo "========================================="
echo "Completed: $(date)"
echo ""

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "All 5 tasks completed successfully!"
else
    echo "Failed tasks (${#FAILED_TASKS[@]}/5):"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
fi

echo ""
echo "Results saved to:"
echo "  experiments/ablation/*/best_interface.py"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Compare results against full evolution runs"
echo "2. Check best_interface.py and metrics.json in each run directory"
echo "3. Plot comparison graphs"
echo "========================================="
