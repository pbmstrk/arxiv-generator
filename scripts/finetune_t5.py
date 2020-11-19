import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import random_split

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from title_generator import TitleGenerator
from title_generator.encoder import T5Encoder
from title_generator.data import DataModule, ArxivDataset
from title_generator.callbacks import PrintingCallback, GeneratorCallback

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Arguments:\n %s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.random.seed)

    filepath = hydra.utils.to_absolute_path(cfg.dataset.filepath)

    dataset = ArxivDataset(filepath=filepath, max_size=cfg.dataset.max_size)
    train, val = random_split(dataset, [cfg.dataset.train_size, cfg.dataset.val_size], 
                        generator=torch.Generator().manual_seed(42))
    encoder = T5Encoder(cfg.encoder.tokenizer_path)
    dm = DataModule(train=train, collate_fn=encoder.collate_fn, val=val, batch_size=cfg.datamodule.batch_size)

    model = TitleGenerator(cfg.model.model_name)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="./checkpoints/" + "{epoch}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
        callbacks=[PrintingCallback("val_loss"), GeneratorCallback(data=val, encoder=encoder),
        early_stop_callback], **cfg.trainer)

    trainer.fit(model, dm.train_dataloader, dm.val_dataloader)