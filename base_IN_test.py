# %%
%reload_ext autoreload
%autoreload 2
import types
from config import get_config
from dataset import data_loader
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from neural_methods import trainer
from neural_methods.trainer import CNNRNNTrainer

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
args.config_file = "./configs/infer_configs/UBFC-rPPG_UBFC-PHYS_TSCAN_IN.yaml"
config = get_config(args)
print('Configuration:')
print(config, end='\n\n')
data_loader_dict = dict()

# %%
def test(config, data_loader_dict):
    """Tests the model."""
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
        model_trainer = trainer.CNNRNNTrainer.CNNRNNTrainer(config, data_loader_dict) 
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)
# %%
test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
    print("test dataset name :" + str(config.TEST.DATA.DATASET) + "->test_loader (Data Loader)")
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
else:
    data_loader_dict['test'] = None
# %%
test(config, data_loader_dict)
# %%
