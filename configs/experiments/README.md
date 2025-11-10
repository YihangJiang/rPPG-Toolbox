# Experiments Configuration Directory 🧪

This directory contains **minimal experiment configurations** using the auto-loading system.

## 🎯 Two Versions Available

### Version 1: Hierarchical (v1)
- Uses BASE inheritance
- Still includes preprocessing in config (~110 lines)
- Example: `tscan_ubfc_rppg_to_phys.yaml`

### Version 2: Minimal (v2) ⭐ **RECOMMENDED**
- Auto-loads dataset templates
- NO preprocessing in config (~40 lines)
- Example: `tscan_ubfc_rppg_to_phys_v2.yaml`

---

## 🚀 Quick Start (Minimal System)

### 1. Copy the Minimal Template

```bash
cp tscan_ubfc_rppg_to_phys_v2.yaml my_experiment.yaml
```

### 2. Edit Your Experiment (Only 40 lines!)

```yaml
# my_experiment.yaml
BASE: ['../base/models/tscan.yaml']
TOOLBOX_MODE: "train_and_test"

TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 30
  LR: 9e-3
  MODEL_FILE_NAME: my_experiment
  DATA:
    DATASET: UBFC-rPPG  # <-- All preprocessing auto-loaded!
    BEGIN: 0.0          # Only experiment-specific
    END: 0.8

VALID:
  DATA:
    DATASET: UBFC-rPPG
    BEGIN: 0.8
    END: 1.0

TEST:
  DATA:
    DATASET: UBFC-PHYS  # <-- Different dataset, auto-loaded!
    BEGIN: 0.0
    END: 1.0
```

### 3. Run Your Experiment

```python
from config import get_config
from config_helper import apply_dataset_templates  # Import helper
import types

args = types.SimpleNamespace()
args.config_file = "configs/experiments/my_experiment.yaml"
config = get_config(args)

# Auto-load dataset templates
apply_dataset_templates(config)  # <-- Magic happens here!

# Now config.TRAIN.DATA has all preprocessing!
```

Or use the provided script:

```bash
python scripts/tscan_minimal_config_training.py
```

---

## 📚 Available Templates

### Models (choose one)
- `../base/models/tscan.yaml` - TSCAN model
- `../base/models/deepphys.yaml` - DeepPhys model
- `../base/models/physnet.yaml` - PhysNet model
- `../base/models/efficientphys.yaml` - EfficientPhys model
- `../base/models/physformer.yaml` - PhysFormer model

### Datasets (auto-loaded)
- `../base/datasets/ubfc_rppg.yaml` - UBFC-rPPG (auto-loaded when DATASET: UBFC-rPPG)
- `../base/datasets/ubfc_phys.yaml` - UBFC-PHYS (auto-loaded when DATASET: UBFC-PHYS)
- `../base/datasets/pure.yaml` - PURE (auto-loaded when DATASET: PURE)
- `../base/datasets/scamps.yaml` - SCAMPS (auto-loaded when DATASET: SCAMPS)

---

## 💡 Example Patterns

### Cross-Dataset Evaluation
```yaml
TRAIN:
  DATA:
    DATASET: PURE       # Auto-loads PURE preprocessing
TEST:
  DATA:
    DATASET: UBFC-rPPG  # Auto-loads UBFC-rPPG preprocessing
```

### Different Model
```yaml
BASE: ['../base/models/deepphys.yaml']  # Change model
TRAIN:
  DATA:
    DATASET: UBFC-rPPG  # Same dataset
```

### Override Dataset Setting
```yaml
TRAIN:
  DATA:
    DATASET: UBFC-rPPG              # Load template
    DATA_PATH: "/my/custom/path"    # Override just the path
```

---

## 📊 Comparison

| Feature | v1 (Hierarchical) | v2 (Minimal) |
|---------|-------------------|--------------|
| **Lines of code** | ~110 | ~40 |
| **Preprocessing** | Included | Auto-loaded |
| **Data paths** | Hardcoded | Auto-loaded |
| **Maintenance** | Medium | Easy |
| **Duplication** | Some | Zero |
| **Recommended** | ⚠️ | ⭐ YES |

---

## 📖 Documentation

- **`docs/MINIMAL_CONFIG_GUIDE.md`** - Complete guide for v2 system
- **`docs/HIERARCHICAL_CONFIG_SYSTEM.md`** - Original hierarchical system
- **`MINIMAL_CONFIG_SUMMARY.md`** - Quick summary

---

## 🧪 Example Scripts

- **`scripts/tscan_minimal_config_training.py`** - Uses v2 (minimal + auto-loading)
- **`scripts/tscan_hierarchical_training.py`** - Uses v1 (hierarchical)
- **`scripts/compare_minimal_vs_old.py`** - Compare old vs new systems

---

## 🎯 Benefits of Minimal System (v2)

✅ **66% smaller configs** (40 lines vs 119)  
✅ **Zero duplication** (preprocessing defined once)  
✅ **Auto-loading** (write `DATASET: name` and done!)  
✅ **Easy maintenance** (update 1 file, affect all)  
✅ **Clear code** (only see experiment-specific settings)  

---

## 🚀 Naming Convention

Suggested: `{model}_{train_data}_to_{test_data}_v2.yaml`

Examples:
- `tscan_ubfc_rppg_to_phys_v2.yaml` ⭐ Minimal
- `deepphys_pure_to_ubfc_v2.yaml`
- `physnet_scamps_ablation_v2.yaml`

---

**Use v2 (minimal) for new experiments!** 🎉


