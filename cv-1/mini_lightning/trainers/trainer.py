from typing import Optional, Tuple
import torch

from ..losses.loss import Loss

from ..metrics import Metric
from ..loggers.logger import Logger
from ..callbacks.callback import Callback
from ..modules.model_module import ModelModule
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

class Trainer:
    def __init__(self, name: str, max_epochs: int, metrics: list[Metric], losses: list[Loss], logger: Optional[Logger] = None, callbacks: Optional[list[Callback]] = None, limit_train_batches: Optional[int] = None):
        self.limit_train_batches = limit_train_batches or float('inf')
        self.name = name
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []
        self.logger = logger
        self.metrics = metrics
        self.losses = losses


    def fit(self, model_module: ModelModule, train_data_loader: DataLoader, valid_data_loader: DataLoader):
        optimizer = self._on_training_start(model_module)
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(train_data_loader):
                total_loss, losses, metrics = self._on_batch(batch_idx, batch, model_module, optimizer)
                epoch_loss += total_loss
            self._on_epoch_end(epoch, epoch_loss, model_module, train_data_loader, valid_data_loader)    
        self._on_train_end(model_module)


    def _on_training_start(self, model_module):
        optimizer = model_module.configure_optimizers()
        for c in self.callbacks:
            c.on_train_start(self, model_module)
        return optimizer


    def _on_batch(self, batch_idx: int, batch: Tuple[torch.Tensor, torch.Tensor], model_module: ModelModule, optimizer: Optimizer):
        if batch_idx > self.limit_train_batches:
            return
        optimizer.zero_grad()
        total_loss, losses, metrics = model_module.training_step(batch, self.metrics, self.losses)
        total_loss.backward()
        optimizer.step()    
        for c in self.callbacks:
            c.on_batch_end(model_module, batch_idx, total_loss)
        return total_loss, losses, metrics


    def _on_epoch_end(self, epoch: int, epoch_loss: float, model_module: ModelModule, train_dl: DataLoader, valid_dl: DataLoader):
        # Smarter logging
        for c in self.callbacks:
            c.on_epoch_end(model_module, epoch)
        if not self.logger:
            return
        
        for batch in valid_dl:
            total_loss, losses, metrics = model_module.eval(batch, self.metrics, self.losses)
        
        global_step = self._calculate_global_step(epoch, train_dl.batch_size)
        self.logger.log_scalar(metrics={
            m.name: v for m, v in metrics.values()
        }, step=global_step)
        
        

    def _on_train_end(self, model_module):
        for c in self.callbacks:
            c.on_train_end(self, model_module)


    @classmethod
    def _calculate_global_step(cls, epoch: int, batch_size: int):
        return epoch * batch_size
        