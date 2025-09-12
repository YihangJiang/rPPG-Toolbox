import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as improvement.
            verbose (bool): If True, prints early stopping messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, current_loss, epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to {self.best_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopping early at epoch {epoch}.")
