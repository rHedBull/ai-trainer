# Neural Trainer

A Claude Code plugin that serves as a comprehensive neural network training coach, focused on Transformers.

## Features

- **Training guidance** - Best practices for training neural networks with mandatory monitoring
- **Debugging assistance** - Systematic diagnosis of training problems (loss plateau, NaN, gradient issues)
- **Scaling law expertise** - Compute-optimal training decisions based on Chinchilla and Kaplan research
- **Hyperparameter tuning** - Experiment design and hyperparameter search strategies
- **Fine-tuning guidance** - LoRA, QLoRA, RLHF, and DPO configuration

## Installation

```bash
/plugin marketplace add rHedBull/ai-trainer
/plugin install neural-trainer@ai-trainer-marketplace
```

## Skills

| Skill | Description |
|-------|-------------|
| `/neural-trainer:train` | Execute training with mandatory monitoring and best-practice defaults |
| `/neural-trainer:train-debug` | Interactive diagnostic workflow for training problems |
| `/neural-trainer:experiment-plan` | Design rigorous experiments before running them |
| `/neural-trainer:scaling-analysis` | Run scaling experiments to understand model/data/compute relationships |
| `/neural-trainer:hyperparam-sweep` | Systematically search hyperparameter space |
| `/neural-trainer:interpret-curves` | Analyze training logs and explain what's happening |
| `/neural-trainer:monitor` | View training progress in real-time |
| `/neural-trainer:run-validation` | Evaluate model checkpoints |

## Agents

These agents activate automatically based on conversation context:

| Agent | Triggers On |
|-------|-------------|
| **diagnostician** | Loss plateau, NaN, gradient explosion/vanishing, training instability |
| **scaling-advisor** | Scaling laws, compute-optimal training, model size decisions, Chinchilla |
| **optimization-guru** | Learning rate, optimizers, schedules, warmup, weight decay |
| **fine-tuning-advisor** | LoRA, QLoRA, RLHF, DPO, catastrophic forgetting |

## Knowledge Base

The plugin includes comprehensive documentation on:

- **Scaling Laws** - Kaplan and Chinchilla research, compute-optimal ratios
- **Optimizer Guide** - AdamW, SGD, Lion, learning rate schedules
- **Transformer Training** - Architecture considerations, initialization, stability tips
- **Common Failures** - Loss plateau, gradient issues, NaN debugging, overfitting
- **Fine-Tuning Techniques** - Full fine-tuning, LoRA, QLoRA, RLHF, DPO

## Requirements

- Claude Code CLI
- Local GPU for training execution (the plugin runs training via shell commands)

## Target Users

Intermediate to experienced ML practitioners who know the fundamentals but need expert-level guidance on:
- Edge cases and debugging
- Optimization strategies
- Scaling decisions
- Fine-tuning configuration

## License

MIT
