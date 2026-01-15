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
