from typing import Mapping
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from .logger import Logger


class TensorboardLogger(Logger):
    def __init__(self, root_dir):
        super().__init__()
        self.experiment = SummaryWriter(root_dir)

    def log_scalar(self, metrics, step):
        for metric, v in metrics.items():
            self.experiment.add_scalar(metric, v, global_step=step)
    
    def log_hyperparameters(self, hyperparams: Mapping[str, float], experiment_metrics: Mapping[str, float]):
        self.experiment.add_hparams(hparam_dict=hyperparams, metric_dict=experiment_metrics)
    
    def log_image_samples(self, grid_name, samples):
        img_grid = torchvision.utils.make_grid(samples)
        self.experiment.add_image(grid_name, img_grid)
