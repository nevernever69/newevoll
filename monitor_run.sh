#!/bin/bash
# Live monitoring dashboard for ablation runs
# Usage: bash monitor_run.sh experiments/ablation/easy_pickup_no_evolution

RUN_DIR=${1:-"experiments/ablation/easy_pickup_no_evolution"}
LOG_FILE="$RUN_DIR/evolution.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    echo "Usage: bash monitor_run.sh <run_directory>"
    exit 1
fi

while true; do
    clear
    echo "=========================================="
    echo "MDP Discovery Run Monitor"
    echo "=========================================="
    echo "Run: $RUN_DIR"
    echo "Time: $(date)"
    echo ""

    # Count candidates
    TOTAL_CANDIDATES=$(grep -c "Running crash filter" "$LOG_FILE" 2>/dev/null || echo 0)
    PASSED=$(grep -c "Crash filter passed" "$LOG_FILE" 2>/dev/null || echo 0)
    CRASHED=$(grep -c "Candidate crashed" "$LOG_FILE" 2>/dev/null || echo 0)
    TRAINING=$(grep -c "Running short training" "$LOG_FILE" 2>/dev/null || echo 0)
    COMPLETED=$(grep -c "full_train success" "$LOG_FILE" 2>/dev/null || echo 0)

    echo "Candidate Status:"
    echo "  Generated:    $TOTAL_CANDIDATES"
    echo "  Passed:       $PASSED"
    echo "  Crashed:      $CRASHED"
    echo "  Training:     $TRAINING"
    echo "  Completed:    $COMPLETED"
    echo ""

    # Best success rate
    BEST=$(grep "NEW BEST" "$LOG_FILE" | tail -1)
    if [ -n "$BEST" ]; then
        echo "Best Candidate:"
        echo "  $BEST"
        echo ""
    fi

    # Recent activity
    echo "Recent Activity:"
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    echo ""

    # GPU usage
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "  GPU: %s%% | Memory: %sMB / %sMB\n", $1, $2, $3}'
    echo ""

    echo "Press Ctrl+C to exit"
    sleep 5
done
