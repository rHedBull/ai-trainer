---
name: scaling-analysis
description: Run scaling experiments to understand model/data/compute relationships. Use when investigating scaling laws, compute-optimal training, or model size decisions.
---

# Scaling Law Investigation

Empirically test how performance scales with model size, data, or compute.

## Step 1: Define Scaling Dimension
Ask what to scale: model size, dataset size, compute, or multiple dimensions

## Step 2: Design Scale Points
Use at least 4 points spanning 1-2 orders of magnitude
Log-space for data/compute scaling

## Step 3: Configure Runs
Same optimizer settings, same data order, same evaluation set across all points

## Step 4: Execute Runs
Use /train skill for each run, collect final validation loss, training curve, compute used

## Step 5: Analyze Results
- Plot scaling curve with power law fit
- Calculate scaling exponent
- Compare to literature (Kaplan alpha~0.076, Chinchilla alpha~0.34)

## Step 6: Report
Fitted scaling law, comparison to literature, extrapolation predictions, recommendations
