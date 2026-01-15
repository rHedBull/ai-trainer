---
name: run-validation
description: Run model validation on a dataset. Use when testing model performance, comparing checkpoints, or running final evaluation.
---

# Model Validation

Run validation on a trained checkpoint.

## Step 1: Identify Checkpoint
Path to checkpoint, run name + step, or "best"/"latest"

## Step 2: Identify Validation Data
Validation set, test set, custom set, or multiple datasets

## Step 3: Define Metrics
- Standard: loss, perplexity (for LM), accuracy
- Task-specific: BLEU, ROUGE, F1, precision, recall

## Step 4: Run Validation
Load model, set to eval mode, iterate through data, compute metrics

## Step 5: Report Results
Single checkpoint table, multi-checkpoint comparison, cross-dataset evaluation
Compare to baselines, provide recommendations based on results
