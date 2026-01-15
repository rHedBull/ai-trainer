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
