#!/usr/bin/env python3
"""
TSCAN Training Script with Minimal Configuration System

This script demonstrates the MINIMAL configuration approach:
- Experiment configs: ONLY experiment-specific settings (BEGIN/END, hyperparameters)
- Dataset templates: ONLY dataset-specific settings (preprocessing, paths, formats)
- Auto-loading: Dataset settings automatically merge when you specify DATASET name

Example Usage:
    python tscan_hierarchical_training.py

The config file is now TRULY minimal:
    OLD: configs/train_configs/UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_TSCAN_BASIC.yaml (119 lines)
    NEW: configs/experiments/tscan_ubfc_rppg_to_phys.yaml (40 lines!)
"""
# %%
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
# Note: apply_dataset_templates() is now called inside get_config() automatically

import types
from dataset import data_loader
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from neural_methods import trainer
import time

# ============================================================================
# Random Seed Configuration for Reproducibility
# ============================================================================
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create generators for data loaders
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def seed_worker(worker_id):
    """Worker init function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================================
# Load Minimal Configuration
# ============================================================================
args = types.SimpleNamespace()
args.config_file = "../configs/experiments/tscan_ubfc_rppg_to_phys.yaml"

config = get_config(args)
# Note: apply_dataset_templates() is now called inside get_config() 
# to ensure correct EXP_DATA_NAME generation with dataset values

data_loader_dict = dict()
# %%
# ============================================================================
# Training Function
# ============================================================================
def train(config, data_loader_dict):
    """
    Trains the model based on configuration.
    
    Args:
        config: Configuration object
        data_loader_dict: Dictionary containing train/valid/test data loaders
    Returns:
        model_trainer: Trained model trainer object
    """
    print(f"\n🚀 Initializing {config.MODEL.NAME} trainer...")
    
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
        print("✓ TSCAN trainer initialized")
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'cnnrnn':
        model_trainer = trainer.CNNRNNTrainer.CNNRNNTrainer(config, data_loader_dict)
        print("✓ CNNRNN baseline trainer initialized")
    else:
        raise ValueError(f'Model {config.MODEL.NAME} is not supported yet!')
    
    print("\n🎓 Starting training...\n")
    model_trainer.train(data_loader_dict)
    return model_trainer

# ============================================================================
# Data Loader Setup
# ============================================================================
print("📊 Setting up data loaders...")
print(f"   Train Dataset: {config.TRAIN.DATA.DATASET}")
print(f"   Valid Dataset: {config.VALID.DATA.DATASET}")
print(f"   Test Dataset: {config.TEST.DATA.DATASET}")

# Select appropriate data loaders based on config
train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader

# ============================================================================
# Training Data Loader
# ============================================================================
print("\n📦 Creating training data loader...")
train_start_time = time.time()
train_data_loader = train_loader(
    name="train",
    data_path=config.TRAIN.DATA.DATA_PATH,
    config_data=config.TRAIN.DATA
)
data_loader_dict['train'] = DataLoader(
    dataset=train_data_loader,
    num_workers=16,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=train_generator
)
train_end_time = time.time()
train_duration = train_end_time - train_start_time
print(f"   ✓ Training batches: {len(data_loader_dict['train'])}")
print(f"   ⏱️  Time taken: {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")

# ============================================================================
# Validation Data Loader
# ============================================================================
print("\n📦 Creating validation data loader...")
valid_start_time = time.time()
valid_data = valid_loader(
    name="valid",
    data_path=config.VALID.DATA.DATA_PATH,
    config_data=config.VALID.DATA
)
data_loader_dict["valid"] = DataLoader(
    dataset=valid_data,
    num_workers=16,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=general_generator
)
valid_end_time = time.time()
valid_duration = valid_end_time - valid_start_time
print(f"   ✓ Validation batches: {len(data_loader_dict['valid'])}")
print(f"   ⏱️  Time taken: {valid_duration:.2f} seconds ({valid_duration/60:.2f} minutes)")

# ============================================================================
# Test Data Loader
# ============================================================================
print("\n📦 Creating test data loader...")
test_start_time = time.time()
test_data = test_loader(
    name="test",
    data_path=config.TEST.DATA.DATA_PATH,
    config_data=config.TEST.DATA
)
data_loader_dict["test"] = DataLoader(
    dataset=test_data,
    num_workers=16,
    batch_size=config.INFERENCE.BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=general_generator
)
test_end_time = time.time()
test_duration = test_end_time - test_start_time
print(f"   ✓ Test batches: {len(data_loader_dict['test'])}")
print(f"   ⏱️  Time taken: {test_duration:.2f} seconds ({test_duration/60:.2f} minutes)")
# ============================================================================
# Main Training and Testing
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🎬 STARTING TRAINING PIPELINE")
    print("=" * 80)
    
    # Train the model
    model_trainer = train(config, data_loader_dict)
    
    print("\n" + "=" * 80)
    print("🧪 STARTING TESTING PHASE")
    print("=" * 80)
    
    # Test the model
    model_trainer.test(data_loader_dict)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING AND TESTING COMPLETE!")
    print("=" * 80)
    print("\n💡 Benefits of Minimal Config System:")
    print("   ✓ Experiment config reduced from 119 → 40 lines (66% reduction!)")
    print("   ✓ Zero duplication (DRY principle)")
    print("   ✓ Dataset settings defined in ONE place only")
    print("   ✓ Preprocessing auto-loaded (no manual copying)")
    print("   ✓ Easy to maintain and update")
    print("   ✓ Clear separation: dataset vs experiment settings")
    print()
    print("📝 Compare:")
    print("   OLD: Copy 60 lines of preprocessing to every experiment")
    print("   NEW: Write 'DATASET: UBFC-rPPG' (1 line!)")
    print()


