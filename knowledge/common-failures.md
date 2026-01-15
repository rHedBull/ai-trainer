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
