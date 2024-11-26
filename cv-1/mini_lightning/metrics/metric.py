from abc import ABC, abstractmethod
import torch


class Metric(ABC):
    @property
    def name(self):
        return type(self).__class__.__name__

    @abstractmethod
    def calculate(self, ground_truth: torch.Tensor, predictions: torch.Tensor):
        ...
    