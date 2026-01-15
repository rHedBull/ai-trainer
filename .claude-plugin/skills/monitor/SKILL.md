---
name: monitor
description: View training progress in real-time. Use when checking on running training, viewing live metrics, or opening dashboards.
---

# Live Training Monitor

View training progress in real-time.

## Step 1: Identify Logging Backend
File-based, TensorBoard, W&B, or terminal output

## Step 2: Launch Monitor

### Terminal Dashboard (ASCII)
Live updating with step progress, loss, LR, gradient norm, ASCII chart

### TensorBoard
Launch with `tensorboard --logdir ./logs --port 6006`

### Weights & Biases
Get run URL from wandb.init() or list recent runs

### Real-time Plot
Matplotlib animation with auto-refresh

## Step 3: Set Up Alerts
Detect loss spikes, NaN, gradient explosion, overfitting

## Quick Commands
- tail -f training.log
- watch -n 1 nvidia-smi
- tmux combined view
