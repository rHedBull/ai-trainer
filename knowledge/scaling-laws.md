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
