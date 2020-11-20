import logging

from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)

class PrintingCallback(Callback):

    def __init__(self, monitor):
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print("Epoch: %d \t Step %d", epoch, trainer.global_step)
        print("Val loss: %.4f", metrics[self.monitor])

