import torch
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def calculate(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        ...
