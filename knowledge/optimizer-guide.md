# Optimizer and Learning Rate Guide

## AdamW - The Default Choice

**When to use:** Almost always. Default for transformers.

**Key hyperparameters:**
- `lr`: 1e-4 to 1e-3 typical for transformers
- `betas`: (0.9, 0.999) default, (0.9, 0.95) for large models
- `eps`: 1e-8 default, increase to 1e-6 if NaN issues
- `weight_decay`: 0.01 to 0.1 typical

**Scaling with model size:**
| Model Size | Learning Rate | Weight Decay |
|------------|---------------|--------------|
| < 125M     | 1e-3 to 5e-4  | 0.01         |
| 125M - 1B  | 5e-4 to 1e-4  | 0.01 - 0.05  |
| 1B - 10B   | 1e-4 to 3e-5  | 0.05 - 0.1   |
| > 10B      | 3e-5 to 1e-5  | 0.1          |

## Learning Rate Schedules

### Warmup
**Always use warmup.** Prevents early training instability.

```
warmup_steps = min(2000, 0.1 * total_steps)
```

Larger models need more warmup (up to 5-10% of training).

### Cosine Decay (Recommended)
```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * step / total_steps))
```
- Smooth decay, works well empirically
- Set `lr_min` to ~10% of `lr_max` or 0

### Linear Decay
```python
lr = lr_max * (1 - step / total_steps)
```
- Simpler, slightly worse than cosine
- Fine for fine-tuning

### Constant with Decay
- Keep constant LR, decay in final 10-20%
- Good for uncertain training length

## Batch Size and Learning Rate

**Linear scaling rule:**
```
lr_new = lr_base * (batch_new / batch_base)
```

**When to use large batches:**
- Sufficient data (won't overfit)
- Gradient accumulation available
- Training is compute-bound

**Gradient accumulation:**
```python
effective_batch = micro_batch * accumulation_steps * num_gpus
```

## SGD with Momentum

**When SGD beats Adam:**
- Very large batch training (scales better)
- When memory-constrained (no moment buffers)
- Some vision tasks

**Typical config:**
```python
SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)
```

## Newer Optimizers

### Lion (2023)
- Memory-efficient (no second moment)
- Often needs 10x lower LR than Adam
- Good for large models when memory-tight

### Sophia (2023)
- Second-order-ish without full Hessian
- Faster convergence claimed
- Less tested in practice

### 8-bit Adam
- Quantized optimizer states
- Same performance, ~50% less memory
- Use via `bitsandbytes` library

## Debugging Optimizer Issues

**Loss not decreasing:**
1. LR too low -> increase 10x
2. LR too high -> decrease 10x
3. Gradient flow issue -> check norms

**Loss spikes:**
1. LR too high
2. Insufficient warmup
3. Data issue (outlier batch)
4. Numerical instability (check for large values)

**NaN loss:**
1. Increase Adam epsilon to 1e-6
2. Add gradient clipping
3. Check for inf in data
4. Reduce LR
