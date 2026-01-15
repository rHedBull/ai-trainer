# Neural Trainer Plugin Design

A Claude Code plugin that serves as a comprehensive neural network training coach, focused on Transformers, with guidance and executable diagnostics for training, tuning, and debugging.

## Overview

**Purpose:** Expert-level guidance for intermediate to experienced ML practitioners on training neural networks well.

**Core Functions:**
- Training pipeline execution with mandatory monitoring
- Hyperparameter tuning strategies and experiment design
- Training signal debugging and diagnostics
- Monitoring and metrics interpretation
- Scaling law analysis

**Approach:**
- Framework-agnostic (methodology over code specifics)
- Guidance-first with executable diagnostics
- Can run experiments, plot results, calculate metrics
- Local GPU environment (direct shell execution)

**Target Users:** Intermediate to experienced practitioners who know fundamentals but need help with edge cases, optimization, and expert sanity checks.

---

## Plugin Structure

```
neural-trainer/
├── plugin.json              # manifest
├── CLAUDE.md                # plugin instructions
├── skills/                  # user-invoked workflows
│   ├── train.md             # execute training runs
│   ├── train-debug.md       # training diagnostics
│   ├── experiment-plan.md   # experiment design
│   ├── scaling-analysis.md  # scaling law investigation
│   ├── hyperparam-sweep.md  # hyperparameter search
│   ├── interpret-curves.md  # log analysis
│   ├── monitor.md           # live training dashboard
│   └── eval.md              # standalone evaluation
├── agents/                  # autonomous specialists
│   ├── diagnostician.md     # deep-dive debugging
│   ├── scaling-advisor.md   # scaling law expertise
│   ├── optimization-guru.md # optimizer/LR guidance
│   └── fine-tuning-advisor.md # PEFT & adaptation
└── knowledge/               # shared reference docs
    ├── scaling-laws.md
    ├── optimizer-guide.md
    ├── transformer-training.md
    ├── common-failures.md
    └── fine-tuning-techniques.md
```

---

## Skills

### `/train` - Execute Training Run

**Purpose:** Train a model with best-practice defaults and proper monitoring.

**Flow:**
1. Identify model and dataset (from context or ask)
2. Review/suggest training config:
   - Optimizer (AdamW defaults, weight decay)
   - LR schedule (warmup + cosine/linear decay)
   - Batch size, gradient accumulation
   - Checkpointing frequency
3. Set up logging (mandatory - refuses without it)
4. Execute training with live monitoring:
   - Report loss, LR, gradient norms periodically
   - Early warning if anomalies detected (loss spike, NaN)
5. On completion: summarize results, save final checkpoint
6. Suggest next steps (evaluation, more training, hyperparameter changes)

**Smart Defaults:**
- Auto-detect framework from code
- Sensible defaults based on model size (larger model → lower LR, more warmup)
- Auto-calculate gradient accumulation for effective batch size

---

### `/train-debug` - Training Diagnostics Workflow

**Purpose:** Guided diagnostic when training isn't going well.

**Flow:**
1. Ask what symptoms they're seeing (loss plateau, NaN, slow convergence, etc.)
2. Request relevant logs/metrics (loss curves, gradient norms, LR schedule)
3. Run diagnostic checks based on symptoms:
   - Gradient histogram analysis
   - Loss curve pattern matching
   - Learning rate vs loss correlation
4. Present findings with specific recommendations
5. Optionally execute fixes and re-run validation

---

### `/experiment-plan` - Experiment Design Coach

**Purpose:** Help design rigorous experiments before running them.

**Flow:**
1. Ask what hypothesis they're testing
2. Suggest variables to control vs vary
3. Recommend metrics to track
4. Calculate compute budget (runs × time)
5. Output an experiment plan checklist
6. Warn about common pitfalls for that experiment type

---

### `/scaling-analysis` - Scaling Law Investigation

**Purpose:** Empirically test how model/data/compute scaling affects performance.

**Flow:**
1. Define what to scale (model size, data, steps)
2. Generate run configs at multiple scale points
3. Execute runs, collect final losses
4. Plot scaling curves
5. Fit power laws, estimate compute-optimal ratios
6. Compare to known scaling laws (Chinchilla, etc.)

---

### `/hyperparam-sweep` - Hyperparameter Search

**Purpose:** Systematically search hyperparameter space.

**Flow:**
1. Define search space (LR range, batch sizes, warmup ratios, etc.)
2. Choose strategy:
   - Grid search (exhaustive, expensive)
   - Random search (usually sufficient)
   - Bayesian (if Optuna/W&B available)
3. Set budget (max runs, max time)
4. Execute runs, log all results
5. Report best config + visualize parameter importance
6. Show performance vs compute tradeoff

---

### `/interpret-curves` - Log Analysis

**Purpose:** Analyze existing training logs and explain what's happening.

**Flow:**
1. Point to log file/directory/W&B run
2. Parse and plot key metrics
3. Detect patterns:
   - Healthy convergence
   - Plateau (when, at what loss)
   - Instability (spikes, oscillation)
   - Overfitting (train/eval divergence)
4. Provide diagnosis and recommendations
5. Compare to similar successful runs if available

---

### `/monitor` - Live Training Dashboard

**Purpose:** View training progress in real-time.

**Options:**
1. Terminal - ASCII loss curves, live updating
2. TensorBoard - launch and open in browser
3. Plot to file - generate PNG snapshots of curves
4. W&B link - open cloud dashboard

---

### `/eval` - Standalone Evaluation

**Purpose:** Evaluate a checkpoint without training.

**Flow:**
1. Load model checkpoint
2. Run on specified eval set
3. Compute metrics
4. Compare to previous checkpoints or baselines
5. Generate evaluation report

---

## Agents

Spawned automatically when keywords/context match, or invoked by skills for deeper investigation.

### `diagnostician` - Training Problem Solver

**Triggers:** loss plateau, NaN, gradient explosion, not converging, training stuck, instability

**Behavior:**
- Asks targeted questions about symptoms
- Requests specific logs/metrics
- Runs diagnostic scripts (gradient histograms, loss analysis)
- Provides ranked hypotheses with confidence
- Suggests specific fixes, can execute them

---

### `scaling-advisor` - Scaling Law Expert

**Triggers:** scaling, compute-optimal, model size, Chinchilla, how big, how much data

**Behavior:**
- Draws on scaling law knowledge docs
- Helps estimate compute budgets
- Advises on model/data tradeoffs
- Can trigger `/scaling-analysis` for empirical validation

---

### `optimization-guru` - Optimizer & LR Specialist

**Triggers:** learning rate, optimizer, AdamW, warmup, schedule, weight decay, momentum

**Behavior:**
- Deep knowledge of optimizer behavior
- Recommends schedules based on model type/size
- Explains why certain configs work
- Debugs optimizer-specific issues (Adam epsilon, beta values)

---

### `fine-tuning-advisor` - PEFT & Adaptation Expert

**Triggers:** fine-tuning, LoRA, QLoRA, adapter, RLHF, DPO, PEFT

**Behavior:**
- Advises on fine-tuning strategy for use case
- Helps configure LoRA ranks, target modules
- Guides RLHF/DPO setup
- Warns about catastrophic forgetting

---

## Monitoring & Evaluation (Core Requirement)

### Mandatory Logging Layer

Every training run **must** have logging. The plugin refuses to train without it configured.

**Supported Backends:**
1. **File-based** - simple CSV/JSON logs, always works, zero dependencies
2. **TensorBoard** - widely supported, local visualization
3. **Weights & Biases** - cloud-based, rich experiment tracking
4. **Custom callback** - user provides logging function

**Minimum Logged Metrics:**
- Loss (train, eval) every N steps
- Learning rate
- Gradient norm (global)
- Throughput (samples/sec)
- Eval metrics at checkpoints

**Recommended Additional Metrics:**
- Per-layer gradient norms
- Activation statistics (mean, std)
- Memory usage
- Attention entropy (for transformers)

### Eval Integration

`/train` always runs evaluation at checkpoints. User specifies:
- Eval dataset
- Metrics (loss, accuracy, perplexity, custom)
- Frequency (every N steps or epochs)

No eval dataset? Plugin warns and suggests held-out split.

---

## Knowledge Documents

### `knowledge/scaling-laws.md`
- Kaplan scaling laws (2020)
- Chinchilla findings (compute-optimal ratios)
- Data-constrained scaling
- Emergent capabilities thresholds
- Practical implications for different budgets

### `knowledge/optimizer-guide.md`
- AdamW deep-dive (betas, epsilon, weight decay)
- SGD with momentum (when it wins)
- Newer optimizers (Lion, Sophia, Shampoo)
- Learning rate schedules (cosine, linear, warmup strategies)
- Batch size / LR relationship

### `knowledge/transformer-training.md`
- Attention stability (QK normalization, etc.)
- Positional encoding considerations
- Layer norm placement (pre-LN vs post-LN)
- Initialization strategies
- Architecture-specific tips (GPT-style vs encoder)

### `knowledge/common-failures.md`
- Loss plateau causes and fixes
- Gradient explosion/vanishing
- NaN debugging checklist
- Overfitting patterns
- Underfitting patterns
- Data issues that look like model issues

### `knowledge/fine-tuning-techniques.md`
- Full fine-tuning considerations
- LoRA (rank selection, target modules)
- QLoRA specifics
- RLHF pipeline
- DPO as alternative
- Catastrophic forgetting mitigation

---

## Auto-Activation & Integration

### CLAUDE.md Plugin Instructions

```markdown
## Neural Trainer Plugin

You are augmented with deep expertise in neural network training.

### Auto-Activation Rules
- Training problems (loss, gradients, convergence) → spawn diagnostician
- Scaling questions → spawn scaling-advisor
- Optimizer/LR questions → spawn optimization-guru
- Fine-tuning discussion → spawn fine-tuning-advisor

### Before Any Training Run
- Verify logging is configured (refuse without it)
- Confirm eval dataset exists or warn
- Show config summary, ask for confirmation

### When Analyzing Logs
- Always plot before concluding
- Compare to baselines when available
- Give concrete numbers, not vague assessments
```

### Skill ↔ Agent Interaction Example

```
User: "My loss is stuck at 2.3"
      ↓
Claude auto-spawns: diagnostician agent
      ↓
Agent investigates, determines LR issue
      ↓
Agent invokes: /hyperparam-sweep (LR-focused)
      ↓
Runs complete, agent reports findings
```

---

## Implementation Notes

- Framework-agnostic: detect PyTorch, TensorFlow, JAX from imports
- Plotting: use matplotlib for portability, ASCII fallback for terminal
- Logging abstraction: thin wrapper that routes to chosen backend
- Distributed training: light coverage, not primary focus

---

*Design created: 2026-01-15*
