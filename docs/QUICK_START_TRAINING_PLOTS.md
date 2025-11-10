# Quick Start: Training PPG Plots

## How to Use (Simple Version)

### Step 1: Run Your Training Script

**Training PPG plots are now automatic!** Just run your training script as usual:

```bash
python scripts/tscan_face_training.py
```

### Step 2: Find Your Plots

Plots will be saved to:
```
train_runs/exp/ppg_plots/train/epoch_<N>/<video_id>/chunk_<chunk_idx>_PPG_comparison.pdf
```

Example:
```
train_runs/exp/ppg_plots/train/
├── epoch_0/          # First epoch (random predictions)
├── epoch_10/         # After 10 epochs of training
├── epoch_20/         # After 20 epochs
└── epoch_29/         # Last epoch (final trained model)
```

## What You'll See

Each plot shows:
- **Blue line** = Ground truth PPG (actual data)
- **Orange line** = Model prediction
- Better overlap = better model performance!

## When Plots are Created

By default, plots are created at:
- **Epoch 0** - See how bad random initialization is
- **Every 10 epochs** - Monitor training progress  
- **Last epoch** - See final results

## Tips

1. **Compare epochs**: Watch how predictions improve from epoch 0 to last epoch
2. **Check for issues**: If predictions don't improve, something might be wrong with training
3. **Save disk space**: Only first 3 chunks per video are plotted (configurable)
4. **No configuration needed**: Plots are generated automatically - nothing to enable!

## That's It!

You now have automatic visualization of your model's training progress. The plots work exactly like the test plots you already have in `ppg_plots/`, but for training data.

**Note**: This feature is now always enabled for all training runs. No configuration needed!

---

For more details, see:
- Full documentation: `docs/TRAINING_PPG_PLOTS.md`
- Training vs Test comparison: `docs/TRAINING_VS_TEST_PLOTS.md`
- Implementation details: `docs/IMPLEMENTATION_SUMMARY.md`

