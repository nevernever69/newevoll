#!/bin/bash
# Setup mujoco_playground and run Go1 ablation study on EC2

set -e  # Exit on error

echo "=== Setting up MuJoCo Playground ==="

# Navigate to repo root
cd ~/newevoll

# Clone mujoco_playground if not already present
if [ ! -d "mujoco_playground" ]; then
    echo "Cloning mujoco_playground..."
    git clone https://github.com/google-deepmind/mujoco_playground.git
    echo "✓ mujoco_playground cloned"
else
    echo "✓ mujoco_playground already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "=== Running Go1 Diagnostic Test (3 candidates) ==="
echo "This will test if MuJoCo setup works correctly."
echo "Assets will download automatically on first run."
echo ""

# Run diagnostic test first (only 3 candidates, cascade enabled, DEBUG logging)
python run.py --config configs/go1_test_cascade.yaml

echo ""
echo "=== Diagnostic test completed ==="
echo ""
read -p "Did the test work? (Each candidate should take 5+ min, not 0.6s) [y/N]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Running Full Go1 Ablation (30 candidates, cascade DISABLED) ==="
    echo "This will take several hours. Monitor with: tail -f nohup.out"
    echo ""

    # Run full ablation with cascade disabled
    nohup python run.py --config configs/go1_push_recovery_noevo.yaml > go1_ablation.log 2>&1 &

    echo "✓ Go1 ablation started in background (PID: $!)"
    echo "✓ Logs: tail -f go1_ablation.log"
    echo "✓ W&B: Check your W&B dashboard for real-time progress"
else
    echo ""
    echo "Diagnostic test failed. Check logs above for errors."
    echo "Common issues:"
    echo "  - Assets failed to download (check internet connection)"
    echo "  - CUDA/GPU issues (check nvidia-smi)"
    echo "  - Import errors (check mujoco_playground clone)"
fi
