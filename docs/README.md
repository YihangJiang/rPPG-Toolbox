# Training PPG Plotting Documentation

This folder contains documentation for the automatic training PPG plotting feature.

## Quick Access

### 📝 Summary
Overview: **[SUMMARY.md](SUMMARY.md)**
- What was implemented
- How to use (simple)
- Where to find plots
- Quick reference

### 🚀 Getting Started
Start here: **[QUICK_START_TRAINING_PLOTS.md](QUICK_START_TRAINING_PLOTS.md)**
- Simple 2-step guide
- No configuration needed
- Get plots immediately

### 📖 Full Documentation
Complete details: **[TRAINING_PPG_PLOTS.md](TRAINING_PPG_PLOTS.md)**
- How it works
- Output locations
- Plot contents
- Troubleshooting
- Customization options

### ⚖️ Training vs Test
Comparison guide: **[TRAINING_VS_TEST_PLOTS.md](TRAINING_VS_TEST_PLOTS.md)**
- When to use training plots
- When to use test plots
- Directory structures
- Common patterns
- Best practices

### 🔧 Implementation
Technical details: **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
- Code changes
- Architecture
- Performance considerations
- Compatibility

## What Is This Feature?

The training PPG plotting feature automatically generates visualizations comparing:
- **Model predictions** (orange line)
- **Ground truth PPG** (blue line)

During training at epochs: **0, 10, 20, ..., last**

## Where Are Plots Saved?

```
train_runs/exp/ppg_plots/train/epoch_<N>/<video_id>/chunk_<X>_PPG_comparison.pdf
```

## Zero Configuration Required

Just run your training script:
```bash
python scripts/tscan_face_training.py
```

Plots are generated automatically - no setup needed!

## File Overview

| File | Purpose | Audience |
|------|---------|----------|
| [SUMMARY.md](SUMMARY.md) | Quick overview | All users |
| [QUICK_START_TRAINING_PLOTS.md](QUICK_START_TRAINING_PLOTS.md) | Get started in 2 steps | All users |
| [TRAINING_PPG_PLOTS.md](TRAINING_PPG_PLOTS.md) | Complete user guide | All users |
| [TRAINING_VS_TEST_PLOTS.md](TRAINING_VS_TEST_PLOTS.md) | Compare training/test plots | Researchers |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical implementation | Developers |

## Need Help?

1. **Just want to use it?** → [QUICK_START_TRAINING_PLOTS.md](QUICK_START_TRAINING_PLOTS.md)
2. **Having issues?** → See "Troubleshooting" in [TRAINING_PPG_PLOTS.md](TRAINING_PPG_PLOTS.md)
3. **Want to customize?** → See "Customization" in [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
4. **Understanding results?** → [TRAINING_VS_TEST_PLOTS.md](TRAINING_VS_TEST_PLOTS.md)

