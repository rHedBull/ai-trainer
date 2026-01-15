---
name: train
description: Execute a neural network training run with mandatory monitoring and best-practice defaults. Use when user wants to train a model, start training, or run a training job.
---

# Training Run Execution

Execute neural network training with proper monitoring, validation, and best practices.

## Pre-Flight Checklist

Before training, verify these requirements:

### 1. Logging Configuration (MANDATORY)

**Ask user which logging backend to use:**
- File-based (CSV/JSON) - zero dependencies
- TensorBoard - local visualization
- Weights & Biases - cloud tracking

**REFUSE to train without logging configured.** Say: "I need logging configured before training. Which backend would you like to use?"

### 2. Validation Dataset

**Check for validation data.** If missing, warn explicitly:
"No validation dataset specified. Training without validation makes it impossible to detect overfitting. Options:
1. Provide validation dataset path
2. Auto-split training data (80/20)
3. Proceed without validation (not recommended)"

### 3. Training Configuration

Review and confirm these settings:

```yaml
# Model
model_path: [path or identifier]
framework: [auto-detect from imports]

# Data
train_data: [path]
val_data: [path or split]
batch_size: [default based on model size]
gradient_accumulation: [calculate for effective batch]

# Optimizer
optimizer: AdamW
learning_rate: [default based on model size]
weight_decay: 0.01
betas: [0.9, 0.999] or [0.9, 0.95] for large

# Schedule
warmup_steps: [10% of total or 2000, whichever smaller]
lr_schedule: cosine
total_steps: [from epochs * steps_per_epoch]

# Regularization
gradient_clip: 1.0
dropout: [keep model default]

# Checkpointing
checkpoint_every: [every epoch or 1000 steps]
checkpoint_dir: [./checkpoints/run_name]

# Logging
log_every: [every 10-100 steps]
validate_every: [every checkpoint]
```

**Present summary and ask:** "Does this configuration look correct? Any changes?"

## During Training

### Monitoring
Report periodically:
- Current step / total steps
- Train loss (current, moving average)
- Learning rate
- Gradient norm
- Throughput (samples/sec)
- Estimated time remaining

### Early Warnings
Watch for and alert on:
- Loss spike > 2x moving average
- Gradient norm spike > 10x average
- NaN or Inf values
- Validation loss increasing while train decreases (overfitting)
- Learning rate near zero (schedule exhausted)

### Checkpointing
On each checkpoint:
- Save model state
- Save optimizer state
- Run validation
- Log validation metrics
- Compare to best checkpoint

## Post-Training

### Summary Report
```
Training Complete
================
Total steps: X
Final train loss: X.XXX
Best validation loss: X.XXX (step Y)
Total time: Xh Xm
Throughput: X samples/sec

Checkpoints saved to: ./checkpoints/run_name/
Best checkpoint: checkpoint-YYYY.pt
Logs: [tensorboard/wandb/file path]
```

### Recommendations
Based on results, suggest:
- If converged well: "Ready for final testing on held-out test set"
- If overfitting: "Consider early stopping at step X, more data, or regularization"
- If underfitting: "Consider longer training, higher LR, or larger model"
- If unstable: "Review gradient norms, consider lower LR or more warmup"

## Framework-Specific Notes

### PyTorch
```python
# Standard training loop structure
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    scheduler.step()
```

### HuggingFace Transformers
```python
# Use Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[logging_callback]
)
trainer.train()
```

### JAX/Flax
```python
# Functional training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        return model.apply(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```
