from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from ..model_module import ModelModule


class VanilaAutoEncoder(ModelModule):

    def _create_model(self):
        """Creates model for the given module
        """
        ...

    def _configure_optimizer(self) -> optim.Optimizer:
        """Configure optimizers for the model

        Returns:
            Optimizer: pytorch optimizer specifically for this model
        """
        ...

    def forward(self, x) -> Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            Tensor: _description_
        """
        ...