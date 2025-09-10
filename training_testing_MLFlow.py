# %%
# %reload_ext autoreload
# %autoreload 2
import types
from config import get_config
from dataset import data_loader
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from neural_methods import trainer
from neural_methods.trainer import CNNRNNTrainer_MLFlow

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# %%
args = types.SimpleNamespace()
# TSCAN rppg physc
# args.config_file = "/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/configs/train_configs/UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_TSCAN_BASIC.yaml"
# baseline
args.config_file = "/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/configs/train_configs/UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_CNNRNN_WARP.yaml"
config = get_config(args)
print('Configuration:')
print(config, end='\n\n')
data_loader_dict = dict()

# %%
def train(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'cnnrnn':
        model_trainer = trainer.CNNRNNTrainer_MLFlow.CNNRNNTrainer(config, data_loader_dict) 
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    return model_trainer
    
# %%
train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader

# %%
train_data_loader = train_loader(
    name="train",
    data_path=config.TRAIN.DATA.DATA_PATH,
    config_data=config.TRAIN.DATA)
data_loader_dict['train'] = DataLoader(
    dataset=train_data_loader,
    num_workers=16,
    batch_size=config.TRAIN.BATCH_SIZE,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=train_generator
)
# %%

valid_data = valid_loader(
    name="valid",
    data_path=config.VALID.DATA.DATA_PATH,
    config_data=config.VALID.DATA)
data_loader_dict["valid"] = DataLoader(
    dataset=valid_data,
    num_workers=16,
    batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=general_generator
)
# %%

test_data = test_loader(
    name="test",
    data_path=config.TEST.DATA.DATA_PATH,
    config_data=config.TEST.DATA)
data_loader_dict["test"] = DataLoader(
    dataset=test_data,
    num_workers=16,
    batch_size=config.INFERENCE.BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=general_generator
)
# %%
model_trainer = train(config, data_loader_dict)
# %%
model_trainer.test(data_loader_dict)
# %%

