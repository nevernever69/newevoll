#!/bin/bash
# Ablation Comparison: No Evolution vs Full Evolution
# This script runs both conditions for easy_pickup task
# Total budget: 30 candidates each

set -e

# Activate virtual environment
source .venv/bin/activate

# Enable W&B logging
export WANDB_ENABLED=1

echo "========================================"
echo "ABLATION STUDY: Evolution vs No Evolution"
echo "========================================"
echo "Task: easy_pickup"
echo "Budget: 30 candidates each"
echo "No Evolution: 1 iteration × 30 candidates"
echo "Full Evolution: 10 iterations × 3 candidates"
echo "========================================"

# Check if no-evolution run already exists
if [ -d "experiments/ablation/easy_pickup_no_evolution" ]; then
    echo ""
    echo "✓ No Evolution run already exists (skipping)"
    echo "  Success rate: Check W&B dashboard"
else
    echo ""
    echo "[1/2] Running No Evolution baseline..."
    echo "  Config: 1 iteration × 30 candidates (random sampling)"
    python run.py \
        --config configs/easy_pickup_noevo.yaml \
        --run-dir experiments/ablation/easy_pickup_no_evolution

    echo "✓ No Evolution complete"
fi

echo ""
echo "[2/2] Running Full Evolution..."
echo "  Config: 10 iterations × 3 candidates (evolutionary search)"
python run.py \
    --config configs/easy_pickup_fullevo.yaml \
    --run-dir experiments/ablation/easy_pickup_full_evolution

echo ""
echo "========================================"
echo "✓ ABLATION STUDY COMPLETE"
echo "========================================"

# Extract results
echo ""
echo "Results Summary:"
echo "----------------"

if [ -f "experiments/ablation/easy_pickup_no_evolution/controller_state.json" ]; then
    NO_EVO_SUCCESS=$(python -c "import json; print(json.load(open('experiments/ablation/easy_pickup_no_evolution/controller_state.json')))" 2>/dev/null || echo "N/A")
    echo "No Evolution: Check experiments/ablation/easy_pickup_no_evolution/controller_state.json"
fi

if [ -f "experiments/ablation/easy_pickup_full_evolution/controller_state.json" ]; then
    FULL_EVO_SUCCESS=$(python -c "import json; print(json.load(open('experiments/ablation/easy_pickup_full_evolution/controller_state.json')))" 2>/dev/null || echo "N/A")
    echo "Full Evolution: Check experiments/ablation/easy_pickup_full_evolution/controller_state.json"
fi

echo ""
echo "View detailed comparison on W&B:"
echo "https://wandb.ai/nevernever/mdp-discovery"
echo ""
echo "To create a comparison report:"
echo "1. Go to your W&B project"
echo "2. Select both runs (easy_pickup_no_evolution and easy_pickup_full_evolution)"
echo "3. Click 'Compare' to see side-by-side metrics"
echo "4. Key metrics to compare:"
echo "   - best/success_rate (higher is better)"
echo "   - best/obs_dim (observation dimension)"
echo "   - final/total_llm_tokens (cost comparison)"
echo "   - final/wall_time (time comparison)"
