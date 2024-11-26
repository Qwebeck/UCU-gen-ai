import torch
from typing import NamedTuple
from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch import Tensor

from ..losses.loss import Loss

from ..metrics.metric import Metric

class ModelOutput(NamedTuple):
    total_loss: Tensor
    loss_components: dict[Loss, Tensor]
    metrics: dict[Metric, Tensor]

class ModelModule(ABC):

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        """Configure optimizers for the model

        Returns:
            Optimizer: pytorch optimizer specifically for this model
        """

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

    @abstractmethod
    def forward(self, x) -> Tensor:
        ...


