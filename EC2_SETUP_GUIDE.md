# EC2 GPU Setup Guide for MDP Discovery

Complete guide to run the "No Evolution" ablation study on AWS EC2 with GPU.

## Prerequisites

- AWS Account with:
  - EC2 access
  - AWS Bedrock access enabled
  - Claude 4 Sonnet/Claude Sonnet 4.6 model access granted
- SSH key pair for EC2 instance access
- Local copy of this repository

## Part 1: Launch EC2 GPU Instance

### Recommended Instance Types

Choose based on your budget and performance needs:

| Instance Type | GPU | vCPUs | RAM | Cost/hour* | Use Case |
|--------------|-----|-------|-----|-----------|----------|
| g4dn.xlarge | T4 (16GB) | 4 | 16GB | ~$0.50 | Budget, XMinigrid tasks |
| g4dn.2xlarge | T4 (16GB) | 8 | 32GB | ~$0.75 | Better parallelism |
| g5.xlarge | A10G (24GB) | 4 | 16GB | ~$1.00 | Best performance |
| g5.2xlarge | A10G (24GB) | 8 | 32GB | ~$1.50 | MuJoCo tasks |
| p3.2xlarge | V100 (16GB) | 8 | 61GB | ~$3.00 | High performance |

*Approximate US East prices; use Spot instances for 60-70% savings

### Launch Steps

1. **Go to EC2 Console** → Launch Instance

2. **Choose AMI**:
   - Search: "Deep Learning Base OSS Nvidia Driver GPU AMI"
   - Select: Ubuntu 22.04 version
   - This includes pre-installed NVIDIA drivers and CUDA

3. **Instance Configuration**:
   - Instance Type: `g5.xlarge` (recommended) or your choice
   - Key pair: Select or create SSH key
   - Storage: 100GB gp3 EBS volume
   - Security Group: Allow SSH (port 22) from your IP

4. **Launch** and wait for instance to start

5. **Note** your instance's public IP address

### Cost Optimization

For long-running experiments, consider:

- **Spot Instances**: 60-70% cheaper, but can be interrupted
- **Stop vs Terminate**: Stop when not in use (storage costs ~$10/month)
- **CloudWatch Alarms**: Auto-stop idle instances

## Part 2: Initial Setup

### 1. Connect to Instance

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Optional: Set up SSH config for easier access
cat >> ~/.ssh/config << EOF
Host mdp-ec2
    HostName <instance-public-ip>
    User ubuntu
    IdentityFile ~/path/to/your-key.pem
EOF

# Then connect with:
ssh mdp-ec2
```

### 2. Transfer Project Files

Option A: Clone from Git (if pushed to remote):
```bash
git clone <your-repo-url> newevol
cd newevol
```

Option B: Transfer from local machine:
```bash
# From your local machine
scp -i your-key.pem -r /run/media/never/new2/newevol ubuntu@<instance-ip>:~/
```

Option C: Create tar and transfer:
```bash
# On local machine
cd /run/media/never/new2
tar czf newevol.tar.gz newevol/
scp -i your-key.pem newevol.tar.gz ubuntu@<instance-ip>:~/

# On EC2 instance
tar xzf newevol.tar.gz
cd newevol
```

### 3. Run Automated Setup

```bash
cd ~/newevol
bash setup_ec2.sh
```

This script will:
- Update system packages
- Install Python 3.10 and dependencies
- Verify GPU availability
- Install CUDA-enabled JAX
- Create Python virtual environment
- Install project dependencies
- Install AWS CLI

### 4. Configure AWS Credentials

The project uses AWS Bedrock for Claude API access:

```bash
# Configure AWS CLI
aws configure

# Enter your credentials:
# AWS Access Key ID: [Your Key]
# AWS Secret Access Key: [Your Secret]
# Default region: us-east-1
# Default output format: json
```

Verify Bedrock access:
```bash
aws bedrock list-foundation-models --region us-east-1 --query "modelSummaries[?contains(modelId, 'claude')].modelId"
```

### 5. Verify GPU Setup

```bash
# Check GPU
nvidia-smi

# Verify JAX GPU
python3 -c "import jax; print('Devices:', jax.devices())"
# Should output: [cuda(id=0)]
```

## Part 3: Run Ablation Study

### Overview

The "No Evolution" ablation generates 30 independent candidates in a single iteration (no iterative improvement) to prove that evolutionary refinement matters.

**Tasks:**
1. `easy_pickup` - XMinigrid pick-up task
2. `medium_place_near` - XMinigrid placement task
3. `hard_rule_chain` - XMinigrid complex rules
4. `go1_push_recovery` - MuJoCo quadruped stabilization
5. `panda_pick_and_track` - MuJoCo arm tracking

### Automated Run (All 5 Tasks)

```bash
cd ~/newevol
source .venv/bin/activate

# Run all tasks sequentially (recommended for overnight run)
bash run_no_evolution_ablation.sh
```

This will run all 5 tasks sequentially and log results to `experiments/ablation/`.

**Estimated runtime:**
- XMinigrid tasks: 2-4 hours each
- MuJoCo tasks: 4-8 hours each
- **Total: 15-30 hours**

### Manual Run (Individual Tasks)

For more control, run tasks individually:

```bash
source .venv/bin/activate

# XMinigrid Tasks (lighter on GPU)
python run.py --config configs/easy_pickup_noevo.yaml \
    --run-dir experiments/ablation/easy_pickup_no_evolution

python run.py --config configs/medium_place_near_noevo.yaml \
    --run-dir experiments/ablation/medium_place_near_no_evolution

python run.py --config configs/hard_rule_chain_noevo.yaml \
    --run-dir experiments/ablation/hard_rule_chain_no_evolution

# MuJoCo Tasks (GPU-intensive)
python run.py --config configs/go1_push_recovery_noevo.yaml \
    --run-dir experiments/ablation/go1_no_evolution

python run.py --config configs/panda_pick_and_track_noevo.yaml \
    --run-dir experiments/ablation/panda_no_evolution
```

### Running in Background (Recommended)

Use `screen` or `tmux` to keep jobs running after disconnect:

```bash
# Start a screen session
screen -S ablation

# Run your commands
bash run_no_evolution_ablation.sh

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r ablation

# List all sessions
screen -ls
```

Or use `tmux`:
```bash
tmux new -s ablation
bash run_no_evolution_ablation.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t ablation
```

## Part 4: Monitor Progress

### GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# GPU utilization log
nvidia-smi dmon -s u -c 1000 > gpu_usage.log
```

### System Resources

```bash
# CPU and memory
htop

# Disk usage
df -h
```

### Application Logs

```bash
# Follow evolution log
tail -f experiments/ablation/easy_pickup_no_evolution/evolution.log

# Check all ablation logs
ls experiments/ablation/*/evolution.log
```

### Check Results

```bash
# List completed runs
ls experiments/ablation/*/best_interface.py

# View best candidate for a task
cat experiments/ablation/easy_pickup_no_evolution/best_interface.py

# Check metrics
cat experiments/ablation/easy_pickup_no_evolution/metrics.json
```

## Part 5: Retrieve Results

### Option 1: Direct SCP

```bash
# From your local machine
scp -i your-key.pem -r ubuntu@<instance-ip>:~/newevol/experiments/ablation ./results_ablation
```

### Option 2: Tar and Transfer

```bash
# On EC2 instance
cd ~/newevol
tar czf ablation_results.tar.gz experiments/ablation/

# From local machine
scp -i your-key.pem ubuntu@<instance-ip>:~/newevol/ablation_results.tar.gz ./
tar xzf ablation_results.tar.gz
```

### Option 3: S3 Backup (Recommended)

```bash
# On EC2 instance
aws s3 cp experiments/ablation/ s3://your-bucket/mdp-discovery/ablation/ --recursive

# From local machine
aws s3 sync s3://your-bucket/mdp-discovery/ablation/ ./results_ablation/
```

## Part 6: Cost Management

### During Experiments

- Instance runs: ~$1-2/hour (g5.xlarge)
- Bedrock API: ~$3-15/1M input tokens, ~$15-75/1M output tokens
- Storage: ~$0.10/GB-month

**Estimated total cost per ablation run: $100-300** (depends on task complexity)

### After Completion

1. **Stop Instance** (preserves data):
   ```bash
   # From AWS Console or CLI
   aws ec2 stop-instances --instance-ids <instance-id>
   ```

2. **Backup and Terminate** (no ongoing costs):
   ```bash
   # Backup results to S3
   aws s3 sync ~/newevol/experiments/ s3://your-bucket/mdp-experiments/

   # Terminate instance
   aws ec2 terminate-instances --instance-ids <instance-id>
   ```

### Monitoring Costs

- **AWS Cost Explorer**: Track daily spending
- **CloudWatch Billing Alarms**: Alert on budget thresholds
- **Bedrock Usage Dashboard**: Monitor API calls

## Part 7: Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, install drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

### JAX CUDA Issues

```bash
# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### AWS Bedrock Access Denied

1. Check IAM permissions include `bedrock:*`
2. Verify model access in Bedrock console: AWS Console → Bedrock → Model Access
3. Request Claude 4 Sonnet access if not enabled

### Out of Memory

For large tasks:
- Reduce `num_envs` in config
- Use larger instance type (g5.2xlarge)
- Enable `enable_bf16: true` for memory savings

### Task Crashes

Check crash logs:
```bash
tail -n 100 experiments/ablation/*/evolution.log | grep -i error
```

Common issues:
- Syntax errors in generated code (retry will fix)
- Timeout issues (increase `evaluator.timeout`)
- CUDA OOM (reduce `num_envs`)

## Part 8: Next Steps

After ablation completes:

1. **Compare Results**:
   - Check `best_interface.py` in each run directory
   - Compare success rates with full evolution runs
   - Analyze `metrics.json` for detailed stats

2. **Visualization**:
   ```bash
   python plot_ablation_clean.py
   ```

3. **Run Full Evolution** (for comparison):
   ```bash
   # Use original configs with 30 iterations
   python run.py --config configs/easy_pickup.yaml --iterations 30
   ```

4. **Paper Results**:
   - Document success rate differences
   - Plot evolution curves
   - Statistical significance tests

## Quick Reference Commands

```bash
# Setup
bash setup_ec2.sh
aws configure

# Run all ablations
bash run_no_evolution_ablation.sh

# Monitor
watch -n 1 nvidia-smi
tail -f experiments/ablation/*/evolution.log

# Retrieve results
tar czf results.tar.gz experiments/ablation/
scp -i key.pem ubuntu@<ip>:~/newevol/results.tar.gz ./

# Stop instance (AWS Console or CLI)
aws ec2 stop-instances --instance-ids <id>
```

## Support

For issues:
1. Check logs: `experiments/ablation/*/evolution.log`
2. Test GPU: `nvidia-smi` and `python -c "import jax; print(jax.devices())"`
3. Test Bedrock: `aws bedrock list-foundation-models --region us-east-1`
4. Review config: `cat configs/*_noevo.yaml`

---

**Ready to start?** Transfer files → Run `setup_ec2.sh` → Run `run_no_evolution_ablation.sh`
