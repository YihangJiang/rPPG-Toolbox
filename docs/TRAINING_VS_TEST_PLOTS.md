# Training vs Test PPG Plots: A Comparison

## Overview

This document explains the differences between training PPG plots and test PPG plots in the rPPG-Toolbox.

## Side-by-Side Comparison

| Feature | Training Plots | Test Plots |
|---------|---------------|------------|
| **Purpose** | Monitor training progress | Evaluate final model performance |
| **When Generated** | During training (selected epochs) | After training (during testing) |
| **Frequency** | Epochs 0, 10, 20, ..., last | Once (after training complete) |
| **Data Source** | Training dataset | Test dataset |
| **Model State** | Varies by epoch | Best or last epoch |
| **Location** | `<model_dir>/ppg_plots/train/epoch_<N>/` | `<test_path>/<exp_name>/ppg_plots/` |
| **Config Option** | Always automatic | Always automatic |
| **Chunks Plotted** | First 3 per video | All chunks |
| **Primary Use** | Debugging, monitoring | Final evaluation, publication |

## Directory Structure Comparison

### Training Plots
```
train_runs/exp/
└── ppg_plots/
    └── train/
        ├── epoch_0/
        │   ├── subject01/
        │   │   ├── chunk_0_PPG_comparison.pdf
        │   │   ├── chunk_1_PPG_comparison.pdf
        │   │   └── chunk_2_PPG_comparison.pdf
        │   └── subject02/
        │       └── ...
        ├── epoch_10/
        └── epoch_29/
```

### Test Plots
```
test_runs/exp/
└── UBFC-PHYS/
    └── ppg_plots/
        ├── Best_Tscan_UBFC-PHYS_PPG_comparison.pdf
        ├── Best_Tscan_UBFC-PHYS_PPG_comparison_zoomed.pdf
        ├── subject01/
        │   ├── chunk_0_PPG_comparison.pdf
        │   ├── chunk_1_PPG_comparison.pdf
        │   ├── chunk_2_PPG_comparison.pdf
        │   ├── chunk_3_PPG_comparison.pdf
        │   └── ... (all chunks)
        └── subject02/
            └── ... (all chunks)
```

## Use Cases

### When to Use Training Plots

1. **During Development**
   - Debugging training issues
   - Checking if model is learning
   - Identifying overfitting early
   - Tuning hyperparameters

2. **Progress Monitoring**
   - Visual confirmation of training progress
   - Comparing early vs late epochs
   - Sharing updates with team

3. **Problem Diagnosis**
   - Signal quality issues
   - Data preprocessing problems
   - Model architecture issues

### When to Use Test Plots

1. **Final Evaluation**
   - Assessing final model performance
   - Comparing different models
   - Publication figures

2. **Performance Analysis**
   - Detailed per-subject analysis
   - Identifying failure cases
   - Computing metrics on specific segments

3. **External Validation**
   - Testing on unseen data
   - Cross-dataset evaluation
   - Generalization assessment

## Example Workflow

### Typical Training + Testing Workflow

1. **Start Training**
   ```bash
   python scripts/tscan_face_training.py
   ```
   Plots will be generated automatically at epochs 0, 10, 20, ..., last

2. **Monitor Training (Epochs 0, 10, 20, 29)**
   - Check `train_runs/exp/ppg_plots/train/epoch_0/` → Random predictions
   - Check `train_runs/exp/ppg_plots/train/epoch_10/` → Learning started?
   - Check `train_runs/exp/ppg_plots/train/epoch_20/` → Improving?
   - Check `train_runs/exp/ppg_plots/train/epoch_29/` → Good fit?

3. **Run Testing**
   ```python
   model_trainer.test(data_loader_dict)
   ```

4. **Analyze Test Results**
   - Check `test_runs/exp/UBFC-PHYS/ppg_plots/` for all test plots
   - Compare with best training epoch plots
   - Verify generalization to test set

5. **Compare Training vs Test**
   - Are test predictions as good as training predictions?
   - If not → potential overfitting
   - If yes → good generalization!

## Plot Content Comparison

### Training Plot Title
```
Epoch 10 - train - Video subject01 - Chunk 0 - Predicted vs Ground Truth PPG
```

### Test Plot Title
```
Video subject01 - Chunk 0 - Predicted vs Ground Truth PPG
```

## Configuration Examples

### Training and Testing (Default)
```yaml
TOOLBOX_MODE: "train_and_test"
# Both training and test plots are generated automatically
```

### Testing Only (Final Evaluation)
```yaml
TOOLBOX_MODE: "only_test"
INFERENCE:
  MODEL_PATH: "train_runs/exp/Best_Tscan.pth"
# Only test plots will be generated (no training)
```

## Performance Impact

### Training Plots
- **Time**: ~1-2 minutes per plotting epoch
- **Disk**: ~50-100 MB for full training (30 epochs)
- **Memory**: Minimal (plots freed immediately)

### Test Plots
- **Time**: ~2-5 minutes (plots all chunks)
- **Disk**: ~100-200 MB (more chunks)
- **Memory**: Minimal

## Tips for Using Both

1. **Use training plots for development**
   - Quick feedback during training
   - Iterate on preprocessing/architecture
   - Spot issues early

2. **Use test plots for final validation**
   - Complete evaluation
   - Publication-ready figures
   - Detailed analysis

3. **Compare both for insights**
   - Training good + test bad = overfitting
   - Training bad + test bad = underfitting or data issues
   - Training good + test good = success! 🎉

4. **Adjust plotting frequency**
   - More frequent during early development
   - Less frequent for long training runs
   - Always plot first and last epoch

## Common Patterns

### Pattern 1: Successful Training
```
Epoch 0:  Random predictions, no correlation
Epoch 10: Some structure emerging
Epoch 20: Good correlation, minor errors
Last:     Excellent fit

Test:     Similar quality to last training epoch ✓
```

### Pattern 2: Overfitting
```
Epoch 0:  Random predictions
Epoch 10: Improving
Epoch 20: Excellent fit on training
Last:     Near-perfect on training

Test:     Much worse than training epoch ✗
```

### Pattern 3: Not Learning
```
Epoch 0:  Random predictions
Epoch 10: Still random
Epoch 20: Still random
Last:     Still random ✗

→ Check data, model, learning rate
```

## Summary

- **Training plots** = Development tool (monitoring, debugging)
- **Test plots** = Evaluation tool (final performance, publication)
- **Use both** for complete understanding of model behavior
- **Training plots** help you get to good **test plots** faster!

## Related Documentation

- Training plots details: `docs/TRAINING_PPG_PLOTS.md`
- Quick start: `QUICK_START_TRAINING_PLOTS.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`

