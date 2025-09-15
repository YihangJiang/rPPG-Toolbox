import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from neural_methods.trainer.EarlyStopping import EarlyStopping
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.CNNRNN import CNNRNNModel
from evaluation.metrics import calculate_metrics  # assuming same path as in TscanTrainer


class CNNRNNTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.early_stopping = EarlyStopping(
            patience = self.config.TRAIN.EARLY_STOPPING_PATIENCE,
            min_delta = 0.01,
            verbose=False
        )

        if config.TOOLBOX_MODE == "train_and_test":
            self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = CNNRNNModel(self.chunk_len).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=3e-4)
            if config.TRAIN.SCHEDULER == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer, factor=1.0, total_iters=self.num_train_batches * config.TRAIN.EPOCHS
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
            self.model = CNNRNNModel(self.chunk_len).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))
        else:
            raise ValueError("CNNRNNTrainer initialized in incorrect toolbox mode!")

        self.writer = SummaryWriter(log_dir = os.path.join(self.model_dir, "board"))

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                lrs.append(self.scheduler.get_last_lr())
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            mean_training_losses.append(np.mean(train_loss))
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                self.writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
                self.writer.add_scalar('Loss/valid', valid_loss, epoch)
                self.early_stopping(valid_loss, epoch)
                print('validation loss:', valid_loss)

                if self.early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}. Best epoch was {self.early_stopping.best_epoch}")
                    break

                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")
            


        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"Best trained epoch: {self.best_epoch}, Min val loss: {self.min_valid_loss}")
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

        self.writer.close()

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("\n===Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_batch in vbar:
                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                pred = self.model(data)
                loss = self.criterion(pred, labels)
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())
        return np.mean(valid_loss)

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("\n===Loading chunks of testing data===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Model path for inference is incorrect.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!" + str(self.config.INFERENCE.MODEL_PATH))
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
                self.model.load_state_dict(torch.load(path))
                print("Testing uses last epoch.")
            else:
                path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
                self.model.load_state_dict(torch.load(path))
                print("Testing uses best model.")

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels_ = test_batch[0].to(self.device), test_batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels_ = labels_.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels_ = labels_[:(N * D) // self.base_len * self.base_len]
                pred = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_ = labels_.cpu()
                    pred = pred.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions:
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config)

        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self.model.state_dict(), path)
        print('Saved Model Path:', path)
