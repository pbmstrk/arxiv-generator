import logging

import numpy as np
from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)


class GeneratorCallback(Callback):
    def __init__(self, data, encoder, predict_abstract=False):
        self.data = data
        self.encoder = encoder
        self.predict_abstract = predict_abstract

    def on_validation_end(self, trainer, pl_module):
        data_len = len(self.data)
        idx = np.random.randint(low=0, high=data_len)
        data_element = self.data[idx]
        abstract, title = data_element["abstract"], data_element["title"]
        log.info("Abstract:\n %s", abstract)
        log.info("Title:\n %s", title)

        inputs = self.encoder(src_texts=title if self.predict_abstract else abstract, return_tensors="pt")
        outputs = pl_module.generate(**inputs)
        string = self.encoder.batch_decode(outputs, skip_special_tokens=True)
        log.info("Model Output:\n %s", string[0])
