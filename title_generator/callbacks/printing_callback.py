import logging

from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)

class PrintingCallback(Callback):

    def __init__(self, monitor):
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print("Epoch: %s \t Step %s", epoch, trainer.global_step)
        print("Val loss: %s", metrics[self.monitor])

