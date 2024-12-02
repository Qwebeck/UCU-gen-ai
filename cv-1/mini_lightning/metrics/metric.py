from abc import ABC, abstractmethod
import torch


class Metric(ABC):
    
    def __str__(self):
        return type(self).__class__.__name__.lower()

    @abstractmethod
    def calculate(self, ground_truth: torch.Tensor, predictions: torch.Tensor):
        ...
    