import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration


class TitleGenerator(pl.LightningModule):
    def __init__(self, model_name="t5-small"):

        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, return_dict=True
        )

    def forward(self, input_ids, **kwargs):

        outputs = self.model(input_ids, **kwargs)
        return outputs

    def step(self, batch, batch_idx, prefix=""):

        src_ids, src_mask, tgt_ids = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        outputs = self(src_ids, attention_mask=src_mask, labels=tgt_ids)
        self.log(prefix+"loss", outputs.loss)
        return outputs.loss

    def training_step(self, batch, batch_idx):

        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):

        return self.step(batch, batch_idx, prefix="val_")

    def generate(self, input_ids, **kwargs):
        input_ids = input_ids.to(self.device)
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        return self.model.generate(input_ids, **kwargs)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        return optimizer
