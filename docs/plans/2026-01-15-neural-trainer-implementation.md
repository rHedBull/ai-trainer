# Neural Trainer Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code plugin that serves as a comprehensive neural network training coach with skills, agents, and knowledge documents.

**Architecture:** Skill-centric design with auto-triggered agents. Skills handle user-invoked workflows (`/train`, `/train-debug`, etc.). Agents spawn automatically on keyword matches for open-ended guidance. Knowledge docs provide the foundation both draw from.

**Tech Stack:** Claude Code plugin system (YAML frontmatter, Markdown content), shell scripts for training execution, Python for analysis/plotting.

---

## Phase 1: Plugin Scaffold

### Task 1: Create plugin.json manifest

**Files:**
- Create: `plugin.json`

**Step 1: Create the manifest file**

```json
{
  "name": "neural-trainer",
  "version": "0.1.0",
  "description": "Comprehensive neural network training coach for Transformers - training, tuning, debugging, and scaling guidance",
  "author": "hendrik",
  "skills": "skills/",
  "agents": "agents/",
  "keywords": ["ml", "training", "transformers", "deep-learning", "pytorch", "fine-tuning"]
}
```

**Step 2: Commit**

```bash
git add plugin.json
git commit -m "feat: add plugin.json manifest"
```

---

### Task 2: Create directory structure

**Files:**
- Create: `skills/.gitkeep`
- Create: `agents/.gitkeep`
- Create: `knowledge/.gitkeep`

**Step 1: Create directories**

```bash
mkdir -p skills agents knowledge
touch skills/.gitkeep agents/.gitkeep knowledge/.gitkeep
```

**Step 2: Commit**

```bash
git add skills agents knowledge
git commit -m "feat: add plugin directory structure"
```

---

### Task 3: Create CLAUDE.md plugin instructions

**Files:**
- Create: `CLAUDE.md`

**Step 1: Write plugin instructions**

```markdown
# Neural Trainer Plugin

You are augmented with deep expertise in neural network training, focused on Transformers.

## Knowledge Base

Reference these knowledge documents for authoritative information:
- `knowledge/scaling-laws.md` - Scaling law research and compute-optimal training
- `knowledge/optimizer-guide.md` - Optimizers, learning rates, schedules
- `knowledge/transformer-training.md` - Transformer-specific training guidance
- `knowledge/common-failures.md` - Debugging training problems
- `knowledge/fine-tuning-techniques.md` - LoRA, QLoRA, RLHF, DPO

## Auto-Activation Rules

Spawn agents automatically based on context:

| Context | Agent |
|---------|-------|
| Loss plateau, NaN, gradient issues, not converging | diagnostician |
| Scaling, compute-optimal, model size, Chinchilla | scaling-advisor |
| Learning rate, optimizer, warmup, schedule | optimization-guru |
| Fine-tuning, LoRA, QLoRA, RLHF, DPO | fine-tuning-advisor |

## Training Run Requirements

**Before ANY training run:**
1. Verify logging is configured - REFUSE to train without monitoring
2. Confirm evaluation dataset exists or explicitly warn user
3. Show config summary, get user confirmation

**Supported logging backends:**
- File-based (CSV/JSON) - always works
- TensorBoard - local visualization
- Weights & Biases - cloud tracking
- Custom callback

**Minimum logged metrics:**
- Train/validation loss every N steps
- Learning rate
- Gradient norm (global)
- Throughput (samples/sec)

## Analysis Standards

When analyzing training logs:
- Always plot before concluding
- Compare to baselines when available
- Give concrete numbers, not vague assessments
- Provide ranked hypotheses with confidence levels
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: add CLAUDE.md plugin instructions"
```

---

## Phase 2: Knowledge Documents

### Task 4: Create scaling-laws.md

**Files:**
- Create: `knowledge/scaling-laws.md`

**Step 1: Write scaling laws knowledge**

```markdown
# Scaling Laws for Neural Language Models

## Kaplan et al. (2020) - Original Scaling Laws

**Key findings:**
- Loss scales as power law with model size, dataset size, and compute
- L(N) is proportional to N^(-0.076) for model parameters
- L(D) is proportional to D^(-0.095) for dataset tokens
- L(C) is proportional to C^(-0.050) for compute (FLOPs)

**Implications:**
- Larger models are more sample-efficient
- Given fixed compute, prefer larger models trained on less data
- Returns diminish but never plateau within tested range

## Chinchilla (Hoffmann et al., 2022) - Compute-Optimal Training

**Key revision:** Kaplan under-trained models. Optimal scaling keeps model and data in balance.

**Compute-optimal ratios:**
- For compute budget C, optimal model size N is approximately C^0.5
- Optimal tokens D is approximately 20 times N (roughly 20 tokens per parameter)
- Previous practice: ~1-2 tokens per parameter (severely undertrained)

**Practical guidelines:**

| Model Size | Optimal Tokens | Approximate Compute |
|------------|----------------|---------------------|
| 125M       | 2.5B           | ~10^18 FLOPs        |
| 350M       | 7B             | ~10^19 FLOPs        |
| 1.3B       | 26B            | ~10^20 FLOPs        |
| 7B         | 140B           | ~10^21 FLOPs        |
| 70B        | 1.4T           | ~10^23 FLOPs        |

## Data-Constrained Scaling

When data is limited (can't reach 20x tokens):
- Multiple epochs acceptable but returns diminish
- After ~4 epochs, effective new data drops significantly
- Consider data augmentation, synthetic data
- Or accept smaller model is compute-optimal for your data

## Emergent Capabilities

Some capabilities appear suddenly at scale:
- Few-shot learning improves smoothly
- Chain-of-thought reasoning: emerges ~100B parameters
- Complex reasoning tasks: threshold varies

**Warning:** Don't rely on emergence for critical capabilities - test explicitly.

## Practical Decision Framework

**Given compute budget:**
1. Estimate total FLOPs available
2. Calculate optimal N from Chinchilla
3. Check if you have ~20N tokens
4. If data-constrained, solve for smaller N where D = 20N

**Given model size:**
1. Target ~20 tokens per parameter
2. Calculate required compute
3. If compute-constrained, consider smaller model

**Given dataset size:**
1. Optimal model is approximately D/20 parameters
2. Training longer (more epochs) gives diminishing returns
```

**Step 2: Commit**

```bash
git add knowledge/scaling-laws.md
git commit -m "docs: add scaling laws knowledge document"
```

---

### Task 5: Create optimizer-guide.md

**Files:**
- Create: `knowledge/optimizer-guide.md`

**Step 1: Write optimizer knowledge**

```markdown
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
```

**Step 2: Commit**

```bash
git add knowledge/optimizer-guide.md
git commit -m "docs: add optimizer guide knowledge document"
```

---

### Task 6: Create transformer-training.md

**Files:**
- Create: `knowledge/transformer-training.md`

**Step 1: Write transformer training knowledge**

```markdown
# Transformer Training Guide

## Architecture Considerations

### Layer Normalization Placement

**Pre-LN (Recommended):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```
- More stable training
- Easier to train deep models
- Used by GPT-2, GPT-3, LLaMA

**Post-LN (Original):**
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```
- Can achieve better final performance
- Requires careful LR warmup
- Used by BERT, original Transformer

### Positional Encodings

**Learned (GPT-style):**
- Simple, works well
- Limited to training context length
- Good default choice

**RoPE (Rotary):**
- Better length extrapolation
- Used by LLaMA, modern models
- Recommended for new models

**ALiBi:**
- No position embeddings
- Linear bias in attention
- Good extrapolation properties

## Initialization

### Standard Init
```python
# Embeddings
nn.init.normal_(embed.weight, std=0.02)

# Linear layers
nn.init.normal_(linear.weight, std=0.02)
nn.init.zeros_(linear.bias)

# Output projection (scaled)
nn.init.normal_(output.weight, std=0.02 / sqrt(2 * n_layers))
```

### Why Scaled Output?
- Residual connections accumulate
- Without scaling: activations grow with depth
- Scale by `1/sqrt(2*n_layers)` to maintain variance

## Attention Stability

### Attention Logits
```python
attn = Q @ K.T / sqrt(d_k)  # Scale by sqrt(d_k)
```

**Why scale?** Without scaling, softmax saturates leading to vanishing gradients.

### QK LayerNorm (Optional)
```python
Q = LayerNorm(Q)
K = LayerNorm(K)
```
- Extra stability for very large models
- Used in some >100B models
- Usually unnecessary for <10B

## Training Stability Tips

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Almost always use
- `max_norm=1.0` is safe default
- Monitor clipping frequency (>10% = LR too high)

### Mixed Precision
```python
scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Watch for:**
- Inf/NaN in fp16 are handled automatically by scaler
- If training stalls, check scaler's scale factor

### Activation Checkpointing
- Trade compute for memory
- Re-compute activations in backward pass
- Essential for large models

## Common Transformer Issues

### Attention Entropy Collapse
**Symptom:** All attention weights concentrate on single token
**Cause:** Over-confident model, poor initialization
**Fix:** Add attention dropout, check init scale

### Embedding Instability
**Symptom:** Loss spikes early in training
**Cause:** Large gradients through embeddings
**Fix:** Reduce embedding LR or use separate LR

### Length Generalization Failure
**Symptom:** Model fails on sequences longer than training
**Fix:** Use RoPE or ALiBi, train on varied lengths
```

**Step 2: Commit**

```bash
git add knowledge/transformer-training.md
git commit -m "docs: add transformer training knowledge document"
```

---

### Task 7: Create common-failures.md

**Files:**
- Create: `knowledge/common-failures.md`

**Step 1: Write common failures knowledge**

```markdown
# Common Training Failures and Fixes

## Loss Plateau

### Symptoms
- Loss stops decreasing
- May oscillate around constant value
- Gradient norms non-zero but loss flat

### Diagnostic Steps
1. Plot loss curve - is it truly flat or very slow?
2. Check learning rate - where in schedule?
3. Check gradient norms - are they healthy (0.1-10)?
4. Check validation loss - is it diverging from train?

### Common Causes and Fixes

**Learning rate too low:**
- Gradient norms healthy but loss flat
- Fix: Increase LR 2-10x

**Learning rate too high:**
- Loss oscillates, doesn't settle
- Gradient norms high or variable
- Fix: Decrease LR, add warmup

**Stuck in local minimum:**
- Usually at very low loss values
- Fix: LR warmup restart, try different seed

**Capacity limit reached:**
- Model too small for task
- Fix: Increase model size or simplify task

**Data exhaustion:**
- Model memorized training data
- Validation loss increasing
- Fix: More data, regularization, early stop

## Gradient Explosion

### Symptoms
- Loss suddenly becomes NaN or inf
- Gradient norms spike to very large values
- Often happens in first few steps

### Diagnostic Steps
1. Print gradient norms per layer
2. Check for outliers in data
3. Verify initialization

### Common Causes and Fixes

**Learning rate too high:**
- Fix: Reduce LR, add warmup

**No gradient clipping:**
- Fix: Add `clip_grad_norm_(params, 1.0)`

**Bad initialization:**
- Fix: Use standard init schemes
- Check custom layers

**Numerical overflow in attention:**
- Fix: Ensure attention scaling by sqrt(d_k)

**Data outliers:**
- Fix: Check data preprocessing, normalize

## Gradient Vanishing

### Symptoms
- Loss decreases very slowly or not at all
- Gradient norms near zero
- Deep layers have no gradient

### Common Causes and Fixes

**Too many layers without residuals:**
- Fix: Add skip connections

**Saturated activations:**
- Sigmoid/tanh outputs near plus or minus 1
- Fix: Use ReLU variants, better init

**Improper normalization:**
- Fix: Add LayerNorm, check placement

## NaN Loss

### Immediate Actions
1. Check if NaN in inputs: `torch.isnan(x).any()`
2. Check gradient norms before NaN
3. Check loss value before NaN

### Common Causes and Fixes

**Numerical overflow:**
- Large logits in softmax
- Fix: Temperature scaling, gradient clipping

**Log of zero/negative:**
- Fix: Add epsilon, check cross-entropy inputs

**Division by zero:**
- In normalization layers
- Fix: Add epsilon (increase Adam eps)

**Inf in data:**
- Fix: Validate data pipeline

**Mixed precision issues:**
- Fix: Use GradScaler, increase eps

## Overfitting

### Symptoms
- Train loss decreases, validation loss increases
- Gap grows over training
- Perfect train accuracy, poor validation

### Common Causes and Fixes

**Insufficient data:**
- Fix: More data, data augmentation

**Model too large:**
- Fix: Smaller model, more regularization

**Training too long:**
- Fix: Early stopping based on validation loss

**Regularization techniques:**
- Dropout (0.1-0.3 typical)
- Weight decay (0.01-0.1)
- Label smoothing (0.1 typical)

## Underfitting

### Symptoms
- Both train and validation loss high
- Model never reaches good performance
- Simple patterns not learned

### Common Causes and Fixes

**Model too small:**
- Fix: Increase parameters

**Learning rate too low:**
- Fix: Increase LR

**Not enough training:**
- Fix: Train longer

**Data issues:**
- Labels incorrect
- Preprocessing destroying signal
- Fix: Verify data pipeline

## Data Issues Masquerading as Model Issues

### Shuffling problems
- Loss very spiky
- Fix: Ensure proper shuffling

### Label leakage
- Suspiciously good performance
- Fix: Audit data pipeline

### Preprocessing inconsistency
- Train/validation gap
- Fix: Same preprocessing for both

### Tokenization issues
- OOV tokens, wrong vocab
- Fix: Verify tokenizer config
```

**Step 2: Commit**

```bash
git add knowledge/common-failures.md
git commit -m "docs: add common failures knowledge document"
```

---

### Task 8: Create fine-tuning-techniques.md

**Files:**
- Create: `knowledge/fine-tuning-techniques.md`

**Step 1: Write fine-tuning knowledge**

```markdown
# Fine-Tuning Techniques Guide

## Full Fine-Tuning

**When to use:**
- Sufficient compute and memory
- Large dataset (>100k examples)
- Task very different from pretraining
- Maximum performance needed

**Best practices:**
- Lower LR than pretraining (1e-5 to 5e-5)
- Short warmup (5-10% of steps)
- Monitor for catastrophic forgetting
- Keep validation set from original domain

**Risks:**
- Catastrophic forgetting of pretrained knowledge
- Overfitting on small datasets
- High compute cost

## LoRA (Low-Rank Adaptation)

### Concept
Instead of updating full weight matrix W, learn low-rank decomposition:
```
W_new = W + BA  where B has shape (d, r), A has shape (r, k), r << min(d, k)
```

### Key Parameters

**Rank (r):**
| Task Complexity | Recommended Rank |
|-----------------|------------------|
| Simple (classification) | 4-8 |
| Medium (QA, summarization) | 16-32 |
| Complex (instruction following) | 64-128 |
| Maximum adaptation | 256 |

**Alpha (scaling):**
```
scaling = alpha / rank
```
- Default: `alpha = rank` (scaling = 1)
- Higher alpha = stronger adaptation
- Typical: `alpha = 2 * rank`

**Target modules:**
```python
# Minimum (query, value projections)
target_modules = ["q_proj", "v_proj"]

# Recommended (all attention)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Maximum (attention + MLP)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

More modules = more capacity but slower training.

### Learning Rate for LoRA
- Higher than full fine-tuning: 1e-4 to 3e-4
- LoRA layers only, base frozen

## QLoRA (Quantized LoRA)

### Concept
- Base model in 4-bit quantization
- LoRA adapters in fp16/bf16
- Dramatically reduces memory

### Memory Savings
| Model Size | Full FT (fp16) | LoRA (fp16) | QLoRA (4-bit) |
|------------|----------------|-------------|---------------|
| 7B         | 28GB           | 16GB        | 6GB           |
| 13B        | 52GB           | 28GB        | 10GB          |
| 70B        | 280GB          | 160GB       | 48GB          |

### Best Practices
- Use NF4 quantization (better than INT4)
- Double quantization saves extra memory
- Paged optimizers for memory spikes

### When QLoRA vs LoRA
- **QLoRA:** Memory-constrained, inference will also be quantized
- **LoRA:** Quality-critical, will merge back to fp16

## RLHF (Reinforcement Learning from Human Feedback)

### Pipeline
1. **SFT:** Supervised fine-tune on demonstrations
2. **Reward Model:** Train on preference pairs
3. **PPO:** Optimize policy against reward model

### SFT Stage
- High-quality instruction-response pairs
- 10k-100k examples typically
- Standard fine-tuning applies

### Reward Model
- Same architecture as policy (often)
- Binary classification on preferences
- Critical: diverse, high-quality preferences

### PPO Stage
- KL penalty to prevent reward hacking
- Typical KL coefficient: 0.01-0.1
- Monitor reward vs KL tradeoff

### Common Issues
- **Reward hacking:** Model exploits reward model flaws
- **Mode collapse:** Outputs become repetitive
- **KL explosion:** Model diverges too far

## DPO (Direct Preference Optimization)

### Concept
Skip reward model, directly optimize on preferences using a modified loss function that implicitly learns the reward.

### Advantages over RLHF
- Simpler (no reward model, no RL)
- More stable training
- Often comparable results

### Key Parameter: beta
- Controls deviation from reference model
- Higher beta = stay closer to reference
- Typical: 0.1-0.5

### Best Practices
- Start from good SFT model
- Quality of preferences is critical
- Compare to RLHF on your specific task

## Catastrophic Forgetting

### Prevention Strategies

**Replay buffer:**
- Mix original pretraining data with fine-tuning data
- 1-10% pretraining data typically sufficient

**Elastic weight consolidation:**
- Penalize changes to important weights
- Compute Fisher information on original task

**Lower learning rate:**
- Less aggressive updates = less forgetting
- But also less adaptation

**LoRA/Adapters:**
- Keep original weights frozen
- Naturally prevents forgetting
- Merge cautiously

### Monitoring
- Measure performance on original task during fine-tuning
- Set acceptable degradation threshold
- Early stop if exceeded
```

**Step 2: Commit**

```bash
git add knowledge/fine-tuning-techniques.md
git commit -m "docs: add fine-tuning techniques knowledge document"
```

---

## Phase 3: Skills

### Task 9: Create train.md skill

**Files:**
- Create: `skills/train.md`

**Step 1: Write the train skill**

```markdown
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
```

**Step 2: Commit**

```bash
git add skills/train.md
git commit -m "feat: add /train skill"
```

---

### Task 10: Create train-debug.md skill

**Files:**
- Create: `skills/train-debug.md`

**Step 1: Write the train-debug skill**

```markdown
---
name: train-debug
description: Interactive diagnostic workflow for training problems. Use when training is failing, loss is stuck, gradients explode, NaN occurs, or convergence is poor.
---

# Training Diagnostics Workflow

Systematic debugging when training isn't going well.

## Step 1: Identify Symptoms

**Ask the user:**
"What symptoms are you seeing? (select all that apply)
1. Loss not decreasing / plateau
2. Loss decreasing very slowly
3. Loss oscillating / unstable
4. Loss went to NaN or Inf
5. Gradient explosion (very large gradients)
6. Gradient vanishing (near-zero gradients)
7. Overfitting (train good, validation bad)
8. Training is very slow
9. Other (describe)"

## Step 2: Gather Information

Based on symptoms, request specific information:

### For all issues:
- Training config (LR, optimizer, batch size, schedule)
- Model size (parameters)
- Dataset size (samples)
- Current step / total steps
- Recent loss values (last 100 steps)

### For loss issues:
- Loss curve plot or values
- Learning rate at current step
- Gradient norm history

### For NaN/Inf:
- Exact step where NaN occurred
- Last valid loss value
- Gradient norms before failure
- Any data preprocessing steps

### For overfitting:
- Train vs validation loss curves
- Model capacity vs dataset size
- Regularization settings (dropout, weight decay)

## Step 3: Run Diagnostics

### Gradient Analysis
```python
# Check gradient norms per layer
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.4f}")
```

**Healthy ranges:**
- Global norm: 0.1 - 10
- Per-layer: within 2 orders of magnitude of each other
- Embedding gradients often larger (OK)

### Loss Curve Analysis
```python
import matplotlib.pyplot as plt
import pandas as pd

# Plot loss with moving average
window = 100
smoothed = pd.Series(losses).rolling(window).mean()
plt.plot(losses, alpha=0.3, label='raw')
plt.plot(smoothed, label=f'{window}-step average')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
```

**Patterns to identify:**
- Flat: learning rate issue or capacity limit
- Oscillating: LR too high
- Spike then recover: data outlier
- Spike then NaN: numerical instability
- Diverging train/validation: overfitting

### Learning Rate Analysis
```python
# Plot LR vs Loss correlation
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(steps, losses)
ax1.set_ylabel('Loss')
ax2.plot(steps, learning_rates)
ax2.set_ylabel('Learning Rate')
plt.xlabel('Step')
plt.savefig('lr_loss_correlation.png')
```

## Step 4: Diagnosis

Based on findings, provide ranked hypotheses:

**Format:**
```
Diagnosis
=========

Most likely cause (80% confidence): [cause]
Evidence: [specific observations]
Fix: [specific action]

Alternative hypothesis (15% confidence): [cause]
Evidence: [observations]
Fix: [action]

Also consider (5%): [cause]
```

## Step 5: Apply Fix

Offer to apply the fix:
- "Would you like me to adjust the learning rate to X?"
- "Would you like me to add gradient clipping?"
- "Would you like me to run a short test with these changes?"

After fix, run validation:
- Train for 100-1000 steps
- Compare metrics to before
- Report whether fix helped

## Common Diagnosis Trees

### Loss Plateau
```
Is gradient norm healthy (0.1-10)?
- Yes: LR likely too low, increase 2-10x
- No, very low (<0.01): Vanishing gradients
  - Check activation functions, add skip connections

Is loss at theoretical minimum?
- Yes: Model may be converged (check validation)
- No: Stuck in local minimum, try warmup restart
```

### NaN Loss
```
Did it happen in first few steps?
- Yes: Initialization or LR issue
  - Reduce LR, check init, add warmup
- No: Accumulated instability
  - Check gradient norms before NaN
    - Exploding: Add clipping, reduce LR
    - Normal: Check for log(0) or div/0 in loss
```

### Overfitting
```
Is model much larger than dataset?
- Yes: Reduce model or get more data
- No: Check regularization
  - Current dropout? Weight decay?
  - Try: dropout 0.1-0.3, weight decay 0.01-0.1
```
```

**Step 2: Commit**

```bash
git add skills/train-debug.md
git commit -m "feat: add /train-debug skill"
```

---

### Task 11: Create remaining skills (experiment-plan, scaling-analysis, hyperparam-sweep, interpret-curves, monitor, run-validation)

**Files:**
- Create: `skills/experiment-plan.md`
- Create: `skills/scaling-analysis.md`
- Create: `skills/hyperparam-sweep.md`
- Create: `skills/interpret-curves.md`
- Create: `skills/monitor.md`
- Create: `skills/run-validation.md`

These skills follow similar patterns to the ones above. Each skill has:
- YAML frontmatter with name, description
- Step-by-step workflow
- Code snippets for execution
- Output templates

See the design document for detailed content of each skill.

**Step: Commit all skills**

```bash
git add skills/
git commit -m "feat: add all training skills"
```

---

## Phase 4: Agents

### Task 12: Create diagnostician.md agent

**Files:**
- Create: `agents/diagnostician.md`

**Step 1: Write the diagnostician agent**

```markdown
---
name: diagnostician
description: >
  Deep-dive training problem solver. Use when user mentions: loss plateau,
  loss stuck, NaN loss, gradient explosion, gradient vanishing, not converging,
  training unstable, training failed, training crashed, bad gradients,
  loss spike, diverging, underfitting, overfitting symptoms.
tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

# Training Diagnostician Agent

You are an expert ML training debugger. Your job is to systematically diagnose and fix training problems.

## Approach

1. **Gather symptoms** - Ask specific questions about what's happening
2. **Request evidence** - Get logs, plots, configs
3. **Form hypotheses** - Rank possible causes by likelihood
4. **Test hypotheses** - Run diagnostic scripts
5. **Recommend fixes** - Specific, actionable solutions
6. **Verify fix** - Confirm the problem is resolved

## Diagnostic Scripts

### Gradient Analysis
```python
# Save this and run to analyze gradients
import torch

def analyze_gradients(model):
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            grad_info[name] = {
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'std': g.std().item(),
                'max': g.max().item(),
                'min': g.min().item(),
            }
    return grad_info

# Check for issues
def diagnose_gradients(grad_info):
    issues = []
    norms = [v['norm'] for v in grad_info.values()]

    if max(norms) > 100:
        issues.append("GRADIENT EXPLOSION: max norm > 100")
    if max(norms) < 1e-7:
        issues.append("GRADIENT VANISHING: all norms < 1e-7")
    if max(norms) / (min(norms) + 1e-10) > 1e6:
        issues.append("GRADIENT IMBALANCE: >1e6 ratio between layers")

    return issues
```

### Loss Curve Analysis
```python
import numpy as np

def analyze_loss_curve(losses, window=100):
    losses = np.array(losses)

    # Trend
    recent = losses[-window:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]

    # Stability
    cv = recent.std() / recent.mean()  # coefficient of variation

    # Plateau detection
    is_plateau = cv < 0.01 and abs(slope) < 1e-5

    return {
        'trend': 'decreasing' if slope < -1e-5 else 'increasing' if slope > 1e-5 else 'flat',
        'stability': 'stable' if cv < 0.1 else 'unstable' if cv > 0.3 else 'moderate',
        'is_plateau': is_plateau,
        'recent_mean': recent.mean(),
        'recent_std': recent.std(),
    }
```

## Decision Trees

### Loss Not Decreasing
```
1. Check gradient norms
   - Near zero: Vanishing gradients
     - Check: activation functions, network depth, init
   - Very large: Exploding gradients
     - Check: learning rate, gradient clipping
   - Normal (0.1-10): Optimization issue
     - Check: learning rate, optimizer config

2. Check learning rate
   - Very low (<1e-6): Increase LR
   - Normal: Check if decayed too fast

3. Check data
   - Shuffling? Preprocessing? Labels correct?
```

### NaN Loss
```
1. When did it happen?
   - First step: Init or LR issue
   - After N steps: Accumulated instability

2. Check gradient norms before NaN
   - Exploding: Add clipping, reduce LR
   - Normal: Check loss computation
     - log(0)? Division by zero? Overflow?

3. Check for Inf in data
   - Validate all inputs are finite
```

## Response Format

Always structure your diagnosis as:

```markdown
## Symptoms Observed
[What you see in the data]

## Likely Cause (Confidence: X%)
[Most probable explanation]
Evidence: [What supports this]

## Alternative Causes
- [Other possibility] (Y% likely)
- [Another possibility] (Z% likely)

## Recommended Fix
[Specific action to take]

## Verification
[How to confirm the fix worked]
```

## When to Escalate

If after investigation you cannot determine the cause:
1. Summarize what you've ruled out
2. Suggest the user provide more information
3. Recommend consulting `/train-debug` skill for structured workflow
```

**Step 2: Commit**

```bash
git add agents/diagnostician.md
git commit -m "feat: add diagnostician agent"
```

---

### Task 13: Create remaining agents (scaling-advisor, optimization-guru, fine-tuning-advisor)

**Files:**
- Create: `agents/scaling-advisor.md`
- Create: `agents/optimization-guru.md`
- Create: `agents/fine-tuning-advisor.md`

Each agent follows the same structure:
- YAML frontmatter with name, description (trigger keywords), tools
- Expert knowledge summary
- Decision frameworks
- Code snippets for analysis
- Response format guidelines

See the design document for detailed content.

**Step: Commit all agents**

```bash
git add agents/
git commit -m "feat: add all training agents"
```

---

### Task 14: Final cleanup and verification

**Step 1: Remove .gitkeep files**

```bash
rm -f skills/.gitkeep agents/.gitkeep knowledge/.gitkeep
```

**Step 2: Verify plugin structure**

```bash
find . -type f \( -name "*.md" -o -name "*.json" \) | sort
```

Expected output:
```
./CLAUDE.md
./docs/plans/2026-01-15-neural-trainer-design.md
./docs/plans/2026-01-15-neural-trainer-implementation.md
./plugin.json
./agents/diagnostician.md
./agents/fine-tuning-advisor.md
./agents/optimization-guru.md
./agents/scaling-advisor.md
./knowledge/common-failures.md
./knowledge/fine-tuning-techniques.md
./knowledge/optimizer-guide.md
./knowledge/scaling-laws.md
./knowledge/transformer-training.md
./skills/experiment-plan.md
./skills/hyperparam-sweep.md
./skills/interpret-curves.md
./skills/monitor.md
./skills/run-validation.md
./skills/scaling-analysis.md
./skills/train-debug.md
./skills/train.md
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize plugin structure"
```

---

## Summary

The implementation creates a complete Claude Code plugin with:

**Skills (8):**
- `/train` - Execute training with monitoring
- `/train-debug` - Diagnose training problems
- `/experiment-plan` - Design experiments
- `/scaling-analysis` - Investigate scaling laws
- `/hyperparam-sweep` - Search hyperparameters
- `/interpret-curves` - Analyze training logs
- `/monitor` - Live training dashboard
- `/run-validation` - Run model validation

**Agents (4):**
- `diagnostician` - Training problem solver
- `scaling-advisor` - Scaling law expert
- `optimization-guru` - Optimizer specialist
- `fine-tuning-advisor` - PEFT expert

**Knowledge (5):**
- Scaling laws
- Optimizer guide
- Transformer training
- Common failures
- Fine-tuning techniques

**Total commits:** ~15 incremental commits
