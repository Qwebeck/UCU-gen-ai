from pathlib import Path
from typing import Optional

from ..metrics.metric import Metric
from ..modules.model_module import ModelOutput
from ..trainers import Callback, ModelModule, Trainer

class Checkpoint(Callback):
    def __init__(self, freq, weights_dir: Path, metric_to_track: Optional[Metric]=None):
        self._weights_dir = weights_dir
        self._save_freq = freq
        if not self._weights_dir.exists():
            self._weights_dir.mkdir(parents=True)
        self._metric_to_track = metric_to_track
        self.best_val_metric = None

    def on_epoch_end(self, module: ModelModule, epoch_idx: int, train_output: ModelOutput, val_output: ModelOutput):
        if (epoch_idx + 1) % self._save_freq == 0:
            checkpoint_path = self._weights_dir / f'{type(module).__class__.__name__}_epoch_{epoch_idx}_loss_{train_output.total_loss}.pth'
            metrics = self._format_metrics(epoch_idx, train_output, val_output)

            if self._is_best_model_by_metric_to_track(val_output):
                module.save(self._weights_dir / 'best_model.pth', **metrics)
            module.save(checkpoint_path, **metrics)
            

    def _format_metrics(self, epoch_idx, train_output, val_output):
        return {
                'epoch': epoch_idx,
                'train_total_loss': train_output.total_loss,
                'train_loss_components': {
                    str(k): v for k, v in train_output.loss_components.items()
                },
                'train_metrics': {
                    str(k): v for k, v in train_output.metrics.items()
                },
                'val_total_loss': val_output.total_loss,
                'val_loss_components': {
                    str(k): v for k, v in val_output.loss_components.items()
                },
                'val_metrics': {
                    str(k): v for k, v in val_output.metrics.items()
                }
            }
    
    def _is_best_model_by_metric_to_track(self, val_output: ModelOutput):
        if self._metric_to_track is not None:
            current_val_metric = val_output.metrics[self._metric_to_track]
            return self.best_val_metric is None or current_val_metric < self.best_val_metric

        return False