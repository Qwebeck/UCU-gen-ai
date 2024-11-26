from abc import ABC, abstractmethod
import torch
from typing import Mapping


class Logger(ABC):

    @abstractmethod
    def log_scalar(self, metrics: Mapping[str, torch.Tensor], step: int):
        ...

    @abstractmethod
    def log_hyperparameters(self, hyperparams: Mapping[str, float], experiment_metrics: Mapping[str, float]):
        ...

    @abstractmethod
    def log_image_samples(self, grid_name: str, samples: Mapping[str, torch.Tensor]):
        ...

    