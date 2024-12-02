from abc import ABC
import torch

from ..modules.model_module import ModelModule, ModelOutput

class Callback(ABC):
    
    def on_train_start(self, trainer, module):
        ...

    def on_batch_end(self, module, batch_idx: int, loss: torch.Tensor):
        ...

    def on_epoch_end(self, module: ModelModule, epoch_idx: int, train_output: ModelOutput, val_output: ModelOutput):
        ...

    def on_train_end(self, trainer, module):
        ...

    def on_validation_end(self, module):
        ...