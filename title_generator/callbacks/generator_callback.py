import logging

import numpy as np

from pytorch_lightning.callbacks import Callback


log = logging.getLogger(__name__)

class GeneratorCallback(Callback):

    def __init__(self, data, encoder):
        self.data = data
        self.encoder = encoder

    def on_validation_end(self, trainer, pl_module):
        data_len = len(self.data)
        idx = np.random.randint(low=0, high=data_len)
        abstract, title = self.data[idx]
        log.info("Abstract:\n %s", abstract)
        log.info("Title:\n %s", title)

        inputs = self.encoder(src_texts=abstract, return_tensors="pt")
        outputs = pl_module.generate(**inputs)[0]
        tokens = self.encoder.tokenizer.convert_ids_to_tokens(outputs, skip_special_tokens=True)
        string = self.encoder.tokenizer.convert_tokens_to_string(tokens)
        log.info("Model title:\n %s", string)
