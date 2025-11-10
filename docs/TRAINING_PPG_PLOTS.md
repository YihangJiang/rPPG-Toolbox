# Training PPG Plotting Feature

## Overview

This feature allows you to visualize model predictions vs ground truth PPG signals during the training process. This helps you:

1. **Monitor training progress visually** - See how well the model predictions match the ground truth over epochs
2. **Debug training issues** - Identify if the model is learning properly or if there are issues with the data or model
3. **Compare training performance** - Visualize differences between early and late training epochs

## How It Works

**Training PPG plots are now generated automatically!** No configuration needed.

Simply run your training script and plots will be created at:
- Epoch 0 (initial)
- Every 10th epoch (10, 20, etc.)
- Last epoch (final)

## Output Location

The plots are saved to:
```
<MODEL_DIR>/ppg_plots/train/epoch_<N>/<video_id>/chunk_<chunk_idx>_PPG_comparison.pdf
```

Where:
- `<MODEL_DIR>` is defined in your config (e.g., `train_runs/exp`)
- `<N>` is the epoch number
- `<video_id>` is the video/subject identifier
- `<chunk_idx>` is the chunk index within the video

## Plot Generation Schedule

To balance training speed and visualization needs, plots are generated at:
- **Epoch 0** (first epoch) - See initial random predictions
- **Every 10th epoch** - Monitor training progress
- **Last epoch** - See final training results

This can be customized in the trainer code by modifying the condition in `TscanTrainer.py`:

```python
if epoch == 0 or epoch == self.max_epoch_num - 1 or (epoch > 0 and epoch % 10 == 0):
```

## Plot Contents

Each plot shows:
- **Ground Truth PPG** (blue line) - The actual PPG signal from the dataset
- **Predicted PPG** (orange line) - The model's predicted PPG signal
- **Time axis** - In seconds if sampling frequency (FS) is available, otherwise sample indices
- **Title** - Includes epoch, dataset name, video ID, and chunk index

## Limitations

To prevent generating too many plots and slowing down training:
- Only the **first 3 chunks** of each video are plotted
- Plots are only generated at specific epochs (as described above)

## Implementation Details

### New Functions

1. **`plot_ppg_signals_train()`** in `evaluation/metrics.py`
   - Plots predicted vs ground truth PPG signals for training data
   - Similar to the test plotting function but adapted for training workflow

2. **`evaluate_and_plot_train()`** in `neural_methods/trainer/TscanTrainer.py`
   - Evaluates the entire training dataset in eval mode
   - Collects predictions and labels
   - Calls the plotting function

### Usage in Training Script

The feature is automatically used during training. No code changes or configuration needed in your training scripts.

## Example

```python
# Simply run your training script (e.g., scripts/tscan_face_training.py)
# Plots will be generated automatically at epochs 0, 10, 20, ..., last

model_trainer = train(config, data_loader_dict)
```

## Troubleshooting

### Plots not being generated

1. Verify the epoch number matches the plotting schedule (0, 10, 20, ..., last)
2. Check the model directory for write permissions
3. Ensure you have matplotlib and required dependencies installed

### Out of memory errors

If plotting causes memory issues:
- Reduce the number of chunks plotted (modify `max_chunks` in `plot_ppg_signals_train()`)
- Plot less frequently (modify the epoch condition in the training loop)

### Missing FS (sampling frequency)

If the time axis shows sample indices instead of seconds:
- Add `FS: <sampling_rate>` to your `TRAIN.DATA` section in the config
- Example: `FS: 30` for 30 Hz sampling rate

## Comparison with Test Plots

| Feature | Test Plots | Training Plots |
|---------|-----------|----------------|
| Location | `test_runs/exp/<dataset>/ppg_plots/` | `<model_dir>/ppg_plots/train/epoch_<N>/` |
| When Generated | After testing | During training (selected epochs) |
| All Chunks | Yes | No (limited to first 3 per video) |
| Config Option | Automatic with test | Always automatic |

## Related Files

- `evaluation/metrics.py` - Contains plotting functions
- `neural_methods/trainer/TscanTrainer.py` - Integrates plotting into training

## Related Documentation

- Quick start: `docs/QUICK_START_TRAINING_PLOTS.md`
- Training vs test comparison: `docs/TRAINING_VS_TEST_PLOTS.md`
- Implementation details: `docs/IMPLEMENTATION_SUMMARY.md`

