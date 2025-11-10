# %%
import os
from dataset import data_loader
# %%
from config import get_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/configs/train_configs/UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_TSCAN_BASIC_IN.yaml')
args = parser.parse_args([])  # empty list means no command line args, so use default


# 2. Load config
config = get_config(args)

# %%
# 3. Initialize UBFC-rPPG Loader
dataset = data_loader.UBFCrPPGLoader.UBFCrPPGLoader(
    name="test",
    data_path=config.TRAIN.DATA.DATA_PATH,
    config_data=config.TRAIN.DATA
)

# 4. Print each subject's frames and labels shapes
print(f"Total clips: {len(dataset)}\n")

for idx in range(len(dataset)):
    frames, labels, filename, chunk_id = dataset[idx]
    print(f"Clip {idx}:")
    print(f"  - Subject ID: {filename}")
    print(f"  - Chunk ID: {chunk_id}")
    print(f"  - Frames Shape: {frames.shape}")  # Typically (C, T, H, W)
    print(f"  - Labels Shape: {labels.shape}")  # Typically (T,)
    print("-" * 40)
# %%

#i like cats