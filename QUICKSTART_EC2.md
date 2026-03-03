# Quick Start: Run "No Evolution" Ablation on EC2

TL;DR guide to get the ablation study running on AWS EC2 GPU instance.

## 1. Launch EC2 Instance (5 minutes)

**Instance Type:** `g5.xlarge` (A10G GPU, 4 vCPUs, 16GB RAM, ~$1/hour)

**AMI:** Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)

**Storage:** 100GB gp3

**Security:** Allow SSH from your IP

## 2. Transfer Files (5 minutes)

```bash
# From your local machine
cd /run/media/never/new2
tar czf newevol.tar.gz newevol/
scp -i your-key.pem newevol.tar.gz ubuntu@<instance-ip>:~/

# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Extract files
tar xzf newevol.tar.gz
cd newevol
```

## 3. Run Setup (10 minutes)

```bash
# Automated setup
bash setup_ec2.sh

# Configure AWS
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1)

# Activate environment
source .venv/bin/activate

# Verify everything works
bash verify_setup.sh
```

## 4. Run Ablation Study (15-30 hours)

```bash
# Use screen to keep running after disconnect
screen -S ablation

# Run all 5 tasks
bash run_no_evolution_ablation.sh

# Detach: Ctrl+A, then D
# Reattach later: screen -r ablation
```

## 5. Monitor Progress

```bash
# GPU usage
watch -n 1 nvidia-smi

# Logs
tail -f experiments/ablation/easy_pickup_no_evolution/evolution.log

# Check results
ls experiments/ablation/*/best_interface.py
```

## 6. Retrieve Results

```bash
# From your local machine
scp -i your-key.pem -r ubuntu@<instance-ip>:~/newevol/experiments/ablation ./results_ablation
```

## 7. Stop Instance

AWS Console → EC2 → Stop Instance (or terminate after backing up)

---

## What Gets Run

The ablation study runs these 5 tasks with **1 iteration × 30 candidates each** (no evolution):

1. **easy_pickup** - XMinigrid pick-up task (~2-3 hours)
2. **medium_place_near** - XMinigrid placement (~3-4 hours)
3. **hard_rule_chain** - XMinigrid complex rules (~4-5 hours)
4. **go1_push_recovery** - MuJoCo quadruped (~5-8 hours)
5. **panda_pick_and_track** - MuJoCo arm tracking (~5-8 hours)

**Total runtime: ~15-30 hours** depending on LLM API latency

**Total cost estimate: $100-300** (instance + Bedrock API)

---

## Troubleshooting

**GPU not detected?**
```bash
nvidia-smi  # Should show GPU
python3 -c "import jax; print(jax.devices())"  # Should show [cuda(id=0)]
```

**AWS Bedrock access denied?**
- Check IAM permissions include `bedrock:*`
- Request Claude model access in Bedrock console

**Out of memory?**
- Use g5.2xlarge instead
- Reduce `num_envs` in config files

---

## Manual Run (Individual Tasks)

If you prefer to run tasks one at a time:

```bash
source .venv/bin/activate

# XMinigrid tasks
python run.py --config configs/easy_pickup_noevo.yaml \
    --run-dir experiments/ablation/easy_pickup_no_evolution

python run.py --config configs/medium_place_near_noevo.yaml \
    --run-dir experiments/ablation/medium_place_near_no_evolution

python run.py --config configs/hard_rule_chain_noevo.yaml \
    --run-dir experiments/ablation/hard_rule_chain_no_evolution

# MuJoCo tasks (GPU-intensive)
python run.py --config configs/go1_push_recovery_noevo.yaml \
    --run-dir experiments/ablation/go1_no_evolution

python run.py --config configs/panda_pick_and_track_noevo.yaml \
    --run-dir experiments/ablation/panda_no_evolution
```

---

**For detailed instructions, see [EC2_SETUP_GUIDE.md](EC2_SETUP_GUIDE.md)**
