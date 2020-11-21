import pytorch_lightning as pl
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM


class Seq2SeqTitleGenerator(pl.LightningModule):
    def __init__(self, 
        model_name, 
        optimizer_name="Adam", 
        optimizer_args={"lr": 3e-5}
    ):

        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, return_dict=True
        )
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args

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
        if hasattr(torch.optim, self.optimizer_name):
            optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), **self.optimizer_args)
        elif hasattr(transformers.optimization, self.optimizer_name):
            optimizer = getattr(transformers.optimization, self.optimizer_name)(self.parameters(), **self.optimizer_args)
        else:
            raise AttributeError(f"{self.optimizer_name} is not a valid name, searched torch.optim and transformers.optimization.")
        return optimizer
