from ..trainers import Callback, ModelModule, Trainer
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def on_train_start(self, trainer: Trainer, module: ModelModule):
        return super().on_train_start(trainer, module)
    
    def on_batch_end(self, module: ModelModule, batch_idx: int, loss):
        return super().on_batch_end(module, batch_idx, loss)
    
    def on_epoch_end(self, module, epoch_idx):
        self.writer.add_scalar()
    
    def on_train_end(self, trainer, module):
        return super().on_train_end(trainer, module)
    
    
