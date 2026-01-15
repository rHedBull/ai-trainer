---
name: interpret-curves
description: Analyze training logs and explain what's happening. Use when reviewing training runs, understanding loss curves, or diagnosing completed training.
---

# Training Curve Analysis

Analyze existing training logs to understand what happened.

## Step 1: Load Logs
Support CSV, JSON, TensorBoard, W&B, or pasted values

## Step 2: Plot Overview
Loss curves (train/val), learning rate, gradient norm in 2x2 grid

## Step 3: Detect Patterns
- Healthy training: steady decrease, train/val tracking, stable gradients
- Loss plateau: detect onset, duration, percentage of training
- Overfitting: detect onset, final gap
- Instability: coefficient of variation, severity
- Divergence: detect onset, preceding signals

## Step 4: Statistical Summary
Overview stats, dynamics assessment, key metrics table, recommendations

## Step 5: Compare Runs (if multiple)
Overlay plots, comparison table with final/best loss and timing
