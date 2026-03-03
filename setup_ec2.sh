#!/bin/bash
# EC2 GPU Instance Setup Script for MDP Discovery
# Usage: bash setup_ec2.sh

set -e  # Exit on error

echo "========================================="
echo "MDP Discovery EC2 GPU Setup"
echo "========================================="

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "Warning: This script is designed for Ubuntu. Continue? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Update system
echo -e "\n[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# 2. Install Python and essential tools
echo -e "\n[2/8] Installing Python 3.10+ and development tools..."

# Detect available Python version
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION="3.11"
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_VERSION="3.10"
    PYTHON_CMD="python3.10"
else
    # Try to install python3.11 or python3.10
    if sudo apt-cache show python3.11 &> /dev/null; then
        PYTHON_VERSION="3.11"
        PYTHON_CMD="python3.11"
    else
        PYTHON_VERSION="3.10"
        PYTHON_CMD="python3.10"
    fi
fi

echo "Using Python ${PYTHON_VERSION}"

# Install Python and tools
sudo apt-get install -y \
    ${PYTHON_CMD} \
    python3-venv \
    python3-pip \
    git \
    htop \
    screen \
    build-essential \
    curl

# Verify Python installation
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    echo "ERROR: Failed to install ${PYTHON_CMD}"
    exit 1
fi

# 3. Verify GPU availability
echo -e "\n[3/8] Verifying GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please use a GPU-enabled AMI with NVIDIA drivers."
    echo "Recommended: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 4. Install CUDA-enabled JAX
echo -e "\n[4/8] Installing CUDA-enabled JAX..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
echo "Detected CUDA version: ${CUDA_VERSION}"

if [[ "$CUDA_VERSION" == "12" ]]; then
    pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [[ "$CUDA_VERSION" == "11" ]]; then
    pip3 install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "Warning: Unsupported CUDA version. Installing default JAX with CUDA 12..."
    pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# 5. Create virtual environment
echo -e "\n[5/8] Creating Python virtual environment..."
cd ~
if [ -d "newevol" ]; then
    cd newevol
else
    echo "ERROR: Project directory 'newevol' not found. Please clone/upload the project first."
    exit 1
fi

${PYTHON_CMD} -m venv .venv
source .venv/bin/activate

# 6. Install project dependencies
echo -e "\n[6/8] Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install --upgrade pip setuptools wheel
    pip install -e .
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: No pyproject.toml or requirements.txt found."
    exit 1
fi

# 7. Install AWS CLI and configure
echo -e "\n[7/8] Installing AWS CLI..."
pip install awscli

echo -e "\nAWS CLI installed. You need to configure it manually:"
echo "  aws configure"
echo "Enter your AWS credentials with Bedrock access."

# 8. Verify JAX GPU setup
echo -e "\n[8/8] Verifying JAX GPU setup..."
python3 -c "import jax; print('JAX devices:', jax.devices()); print('GPU available:', len(jax.devices()) > 0 and 'cuda' in str(jax.devices()[0]))"

# Final instructions
echo -e "\n========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials:"
echo "   aws configure"
echo ""
echo "2. Set environment variables (or create .env file):"
echo "   export AWS_ACCESS_KEY_ID='your-key'"
echo "   export AWS_SECRET_ACCESS_KEY='your-secret'"
echo "   export AWS_DEFAULT_REGION='us-east-1'"
echo ""
echo "3. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "4. Run ablation study:"
echo "   bash run_no_evolution_ablation.sh"
echo ""
echo "5. Or run individual tasks:"
echo "   python run.py --config configs/easy_pickup_noevo.yaml --run-dir experiments/ablation/easy_pickup_no_evolution"
echo ""
echo "========================================="
