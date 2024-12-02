from pathlib import Path
import torch
from typing import Any, NamedTuple, TypeVar, Type
from abc import ABC, abstractmethod
import torch.optim as optim
from torch import Tensor
import torch.nn as nn

from ..losses.loss import Loss

from ..metrics.metric import Metric

T = TypeVar('T', bound='ModelModule')
MODEL_STATE_DICT = 'model_state_dict'
OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
class ModelOutput(NamedTuple):
    total_loss: Tensor
    loss_components: dict[Loss, Tensor]
    metrics: dict[Metric, Tensor]

class ModelModule(ABC):
    def __init__(self):
        self._optimizer = None
        self._model = None
    
    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer:
            return self._optimizer
        return self.configure_optimizers()
    
    @property
    def model(self) -> nn.Module:
        if self._model:
            return self._model
        return self.create_model()
    
    def save(self, path: Path, **kwargs):
        checkpoint = {
            MODEL_STATE_DICT: self.model.state_dict(),
            OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls: Type[T], path: Path) -> tuple[T, dict[str, Any]]:
        checkpoint = torch.load(path)
        model_module = cls()
        model_module.model.load_state_dict(checkpoint[MODEL_STATE_DICT])
        model_module.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
        for key in [MODEL_STATE_DICT, OPTIMIZER_STATE_DICT]:
            checkpoint.pop(key, None)
        return model_module, checkpoint
        
    def configure_optimizers(self):
        self._optimizer = self._configure_optimizer()
        return self._optimizer
    
    def create_model(self):
        self._model = self._create_model()
        return self._model
    
    @abstractmethod
    def _create_model(self):
        """Creates model for the given module
        """
    
    @abstractmethod
    def _configure_optimizer(self) -> optim.Optimizer:
        """Configure optimizers for the model

        Returns:
            Optimizer: pytorch optimizer specifically for this model
        """

    @abstractmethod
    def forward(self, x) -> Tensor:
        ...

        
    def training_step(self, batch, metric_functions: list[Metric], loss_functions: list[Loss]) -> ModelOutput:
        x, y = batch
        preds = self.forward(x)
        losses = {
            l: l.calculate(y, preds) for l in loss_functions
        }
        total_loss = sum(l for l in losses.values())
        metrics = {
            m: m.calculate(y, preds) for m in metric_functions
        }
        return ModelOutput(total_loss, losses, metrics)

    def eval(self, batch, metric_functions: list[Metric], loss_functions: list[Loss]) -> ModelOutput:
        with torch.no_grad():
            return self.training_step(batch, metric_functions, loss_functions)



