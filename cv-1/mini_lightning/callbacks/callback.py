from abc import ABC
import torch

class Callback(ABC):
    
    def on_train_start(self, trainer, module):
        ...

    def on_batch_end(self, module, batch_idx: int, loss: torch.Tensor):
        ...

    def on_epoch_end(self, module, epoch_idx: int):
        ...

    def on_train_end(self, trainer, module):
        ...

    def on_validation_end(self, module):
        ...