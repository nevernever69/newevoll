#!/bin/bash
# Quick fix for Python version issues on EC2
# Run this if setup_ec2.sh failed at Python installation

echo "========================================="
echo "Quick Fix: Python Setup"
echo "========================================="

# Detect what Python version is available
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION="3.11"
    PYTHON_CMD="python3.11"
    echo "Found Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_VERSION="3.10"
    PYTHON_CMD="python3.10"
    echo "Found Python 3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
    echo "Found Python ${PYTHON_VERSION}"
else
    echo "ERROR: No Python 3 found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
    PYTHON_CMD="python3"
fi

echo ""
echo "Using: ${PYTHON_CMD}"
${PYTHON_CMD} --version

# Check if version is sufficient
VERSION_CHECK=$(${PYTHON_CMD} -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)")
if [ "$VERSION_CHECK" != "1" ]; then
    echo "ERROR: Python 3.10+ required, but found:"
    ${PYTHON_CMD} --version
    echo ""
    echo "The Deep Learning AMI should have Python 3.10+."
    echo "Try: ls /usr/bin/python3.*"
    exit 1
fi

# Install venv if needed
echo ""
echo "Installing python3-venv..."
sudo apt-get install -y python3-venv python3-pip

# Create virtual environment
echo ""
echo "Creating virtual environment..."
cd ~/newevol
${PYTHON_CMD} -m venv .venv
source .venv/bin/activate

echo ""
echo "Virtual environment created!"
echo "Python version in venv:"
python --version

# Install dependencies
echo ""
echo "Installing project dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e .

echo ""
echo "========================================="
echo "Setup fixed! Now run:"
echo "  source .venv/bin/activate"
echo "  bash verify_setup.sh"
echo "========================================="
