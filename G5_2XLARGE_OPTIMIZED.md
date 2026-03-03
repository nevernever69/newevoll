# g5.2xlarge Optimized Setup

Configs have been optimized for **g5.2xlarge** instances with **8 vCPUs** and **32GB RAM**.

## Expected Performance

### Time Estimates (Optimized for 2xlarge)

| Task | Time | Speedup vs xlarge |
|------|------|-------------------|
| easy_pickup | **1.5-2h** | 35% faster |
| medium_place_near | **2-2.5h** | 35% faster |
| hard_rule_chain | **2.5-3.5h** | 40% faster |
| go1_push_recovery | **3.5-5h** | 35% faster |
| panda_pick_and_track | **3.5-5h** | 35% faster |
| **TOTAL** | **10-20h** | **35% faster** |

**Previous estimate (g5.xlarge): 15-30 hours**
**New estimate (g5.2xlarge): 10-20 hours** ⚡

## Cost Comparison

| Instance | Time | Cost/hour | Total Cost |
|----------|------|-----------|------------|
| g5.xlarge | 15-30h | $1.00 | $15-30 |
| g5.2xlarge | 10-20h | $1.50 | **$15-30** |

**Result: Same total instance cost, but finishes faster!** 🎉

Plus Bedrock API costs: ~$50-250 (same for both)

## Optimizations Applied

All 5 configs have been updated with:

### 1. Increased Parallel Environments
- **XMinigrid tasks**: `num_envs: 8192 → 12288` (+50%)
- **Go1 (MuJoCo)**: `num_envs: 4096 → 6144` (+50%)
- **Panda (MuJoCo)**: `num_envs: 2048 → 3072` (+50%)

This takes advantage of 8 vCPUs for faster RL training.

### 2. Parallel Candidate Evaluation
- **All tasks**: `parallel_evaluations: 2`

Evaluates 2 candidates simultaneously instead of sequentially. With 30 candidates total, this cuts evaluation time significantly.

### 3. Memory Headroom
With 32GB RAM (vs 16GB), you have plenty of room for:
- Larger batch sizes
- More parallel operations
- No memory-related slowdowns

## Quick Start (g5.2xlarge)

```bash
# 1. Launch g5.2xlarge instance
#    AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
#    Storage: 100GB

# 2. Transfer and setup
scp -i key.pem -r newevol/ ubuntu@<ip>:~/
ssh -i key.pem ubuntu@<ip>
cd newevol
bash setup_ec2.sh

# 3. Configure AWS
aws configure

# 4. Verify (should show 8 CPUs, 1 GPU)
bash verify_setup.sh

# 5. Run ablation (screen recommended for overnight run)
screen -S ablation
source .venv/bin/activate
bash run_no_evolution_ablation.sh

# Detach: Ctrl+A, D
# Reattach: screen -r ablation
```

## Monitoring

```bash
# GPU usage (should be ~90-100% during training)
watch -n 1 nvidia-smi

# CPU usage (should see 8 cores utilized)
htop

# Memory usage (should have plenty of headroom)
free -h

# Progress logs
tail -f experiments/ablation/*/evolution.log
```

## Expected GPU Utilization

With these optimizations:
- **Training phases**: 95-100% GPU utilization
- **Evaluation phases**: 80-95% GPU utilization
- **LLM generation phases**: 0% GPU (waiting for API)

The increased `num_envs` ensures the GPU stays fully utilized during training.

## If You Get OOM Errors

If you somehow run out of memory (unlikely with 32GB):

```bash
# Reduce parallel environments slightly
# Edit the failing config and reduce num_envs by 25%:
num_envs: 12288 → 9216  (for XMinigrid)
num_envs: 6144 → 4608   (for Go1)
num_envs: 3072 → 2304   (for Panda)
```

But with 32GB RAM and these settings, you should be fine!

## Cost Optimization Tips

### Use Spot Instances (60-70% cheaper!)

g5.2xlarge Spot: **~$0.50-0.70/hour** (vs $1.50 on-demand)

**Potential savings: $10-20 total**

Launch as Spot:
```bash
# In EC2 Console: Request Spot Instances
# Or use CLI:
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

**Risk**: Can be interrupted (low risk for 10-20h runs in us-east-1)

### Auto-Stop After Completion

Add to the end of `run_no_evolution_ablation.sh`:

```bash
# Auto-shutdown after completion (optional)
sudo shutdown -h +5  # Shutdown in 5 minutes
```

This gives you time to retrieve results before instance stops.

## What You Get

With g5.2xlarge optimizations:
- ✅ **35% faster** than g5.xlarge
- ✅ **Same total cost** (faster but higher hourly rate)
- ✅ **Better reliability** (more RAM = fewer OOM issues)
- ✅ **Finishes overnight** (10-20h vs 15-30h)

## Next Steps After Run

1. **Check results**:
   ```bash
   ls experiments/ablation/*/best_interface.py
   cat experiments/ablation/*/metrics.json
   ```

2. **Download results**:
   ```bash
   # From local machine
   scp -i key.pem -r ubuntu@<ip>:~/newevol/experiments/ablation ./results
   ```

3. **Compare with full evolution** (if you ran 30-iteration versions earlier)

4. **Stop/terminate instance** to avoid charges

---

**Ready? Transfer files → `bash setup_ec2.sh` → `bash run_no_evolution_ablation.sh`**

**Estimated completion: 10-20 hours from start** ⏱️
