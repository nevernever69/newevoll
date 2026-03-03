#!/bin/bash
# Verification script to check EC2 setup before running ablation study
# Usage: bash verify_setup.sh

echo "========================================="
echo "MDP Discovery Setup Verification"
echo "========================================="
echo ""

ERRORS=0
WARNINGS=0

# Function to check and report
check_pass() {
    echo "✓ $1"
}

check_fail() {
    echo "✗ $1"
    ((ERRORS++))
}

check_warn() {
    echo "⚠ $1"
    ((WARNINGS++))
}

# 1. Check GPU availability
echo "[1/10] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        check_pass "GPU detected: $GPU_NAME ($GPU_MEM)"
    else
        check_fail "nvidia-smi found but not working"
    fi
else
    check_fail "nvidia-smi not found. Need GPU instance with NVIDIA drivers."
fi

# 2. Check Python version
echo "[2/10] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        check_pass "Python $PYTHON_VERSION (>= 3.10 required)"
    else
        check_fail "Python $PYTHON_VERSION found, but 3.10+ required"
    fi
else
    check_fail "Python3 not found"
fi

# 3. Check virtual environment
echo "[3/10] Checking virtual environment..."
if [ -d ".venv" ]; then
    check_pass "Virtual environment exists at .venv/"
    if [ -z "$VIRTUAL_ENV" ]; then
        check_warn "Virtual environment not activated. Run: source .venv/bin/activate"
    else
        check_pass "Virtual environment activated"
    fi
else
    check_fail "Virtual environment not found. Run: python3 -m venv .venv"
fi

# 4. Check JAX installation
echo "[4/10] Checking JAX..."
if python3 -c "import jax" 2>/dev/null; then
    JAX_VERSION=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
    check_pass "JAX installed (version $JAX_VERSION)"

    # Check if JAX can see GPU
    JAX_DEVICES=$(python3 -c "import jax; print(jax.devices())" 2>/dev/null)
    if echo "$JAX_DEVICES" | grep -q "cuda"; then
        check_pass "JAX GPU support enabled: $JAX_DEVICES"
    else
        check_fail "JAX installed but GPU not detected: $JAX_DEVICES"
        echo "   Install CUDA-enabled JAX: pip install 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    fi
else
    check_fail "JAX not installed. Run setup_ec2.sh"
fi

# 5. Check required Python packages
echo "[5/10] Checking Python dependencies..."
REQUIRED_PACKAGES=("boto3" "dacite" "yaml" "numpy" "flax")
ALL_INSTALLED=true

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        continue
    else
        check_fail "Missing Python package: $pkg"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = true ]; then
    check_pass "All required Python packages installed"
fi

# 6. Check AWS CLI
echo "[6/10] Checking AWS CLI..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1 | awk '{print $1}')
    check_pass "AWS CLI installed: $AWS_VERSION"
else
    check_fail "AWS CLI not found. Install: pip install awscli"
fi

# 7. Check AWS credentials
echo "[7/10] Checking AWS credentials..."
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    check_pass "AWS credentials set via environment variables"
elif [ -f ~/.aws/credentials ]; then
    check_pass "AWS credentials configured in ~/.aws/credentials"
else
    check_fail "AWS credentials not found. Run: aws configure"
fi

# 8. Check AWS Bedrock access
echo "[8/10] Checking AWS Bedrock access..."
if command -v aws &> /dev/null; then
    if aws bedrock list-foundation-models --region us-east-1 --output text &> /dev/null; then
        check_pass "AWS Bedrock access verified"

        # Check for Claude models
        CLAUDE_MODELS=$(aws bedrock list-foundation-models --region us-east-1 --query "modelSummaries[?contains(modelId, 'claude')].modelId" --output text 2>/dev/null)
        if [ -n "$CLAUDE_MODELS" ]; then
            check_pass "Claude models available in Bedrock"
        else
            check_warn "Claude models not found. Request access in AWS Bedrock console."
        fi
    else
        check_fail "Cannot access AWS Bedrock. Check IAM permissions and credentials."
    fi
else
    check_warn "Skipping Bedrock check (AWS CLI not available)"
fi

# 9. Check config files
echo "[9/10] Checking ablation config files..."
CONFIGS=(
    "configs/easy_pickup_noevo.yaml"
    "configs/medium_place_near_noevo.yaml"
    "configs/hard_rule_chain_noevo.yaml"
    "configs/go1_push_recovery_noevo.yaml"
    "configs/panda_pick_and_track_noevo.yaml"
)

ALL_CONFIGS_EXIST=true
for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        continue
    else
        check_fail "Config file missing: $config"
        ALL_CONFIGS_EXIST=false
    fi
done

if [ "$ALL_CONFIGS_EXIST" = true ]; then
    check_pass "All 5 ablation config files present"
fi

# 10. Check disk space
echo "[10/10] Checking disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 20 ]; then
    check_pass "Sufficient disk space: ${AVAILABLE_GB}GB available"
else
    check_warn "Low disk space: ${AVAILABLE_GB}GB available (recommend 20GB+)"
fi

# Summary
echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ All checks passed! Ready to run ablation study."
    echo ""
    echo "Next steps:"
    echo "  source .venv/bin/activate"
    echo "  bash run_no_evolution_ablation.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ ${WARNINGS} warning(s) found. You can proceed but review warnings above."
    echo ""
    echo "To run ablation study:"
    echo "  bash run_no_evolution_ablation.sh"
    exit 0
else
    echo "✗ ${ERRORS} error(s) and ${WARNINGS} warning(s) found."
    echo ""
    echo "Please fix errors above before running ablation study."
    echo "Run setup_ec2.sh if you haven't already:"
    echo "  bash setup_ec2.sh"
    exit 1
fi
