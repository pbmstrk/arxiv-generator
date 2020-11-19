from pytorch_lightning.callbacks import Callback

class PrintingCallback(Callback):

    def __init__(self, monitor):
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"Epoch: {epoch} \tStep: {trainer.global_step}")
        print(f"Val loss: {metrics[self.monitor]}")
        
