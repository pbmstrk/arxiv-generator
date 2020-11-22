import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import random_split

from title_generator import Seq2SeqTitleGenerator
from title_generator.callbacks import GeneratorCallback, PrintingCallback
from title_generator.data import ArxivDataset, DataModule
from title_generator.encoder import Seq2SeqTokenizer

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Arguments:\n %s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.random.seed)

    filepath = hydra.utils.to_absolute_path(cfg.dataset.filepath)

    dataset = ArxivDataset(filepath=filepath, max_size=cfg.dataset.max_size,
                        categories=cfg.dataset.categories)
    train, val = random_split(dataset, [cfg.dataset.train_size, cfg.dataset.val_size], 
                        generator=torch.Generator().manual_seed(42))
    encoder = Seq2SeqTokenizer(cfg.encoder.tokenizer_path)
    dm = DataModule(train=train, collate_fn=encoder.collate_fn, val=val, batch_size=cfg.datamodule.batch_size)

    model = Seq2SeqTitleGenerator(
        model_name=cfg.model.model_name,
        optimizer_name=cfg.optimizer.name,
        optimizer_args=cfg.optimizer.args
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=hydra.utils.to_absolute_path(cfg.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
        callbacks=[PrintingCallback("val_loss"), GeneratorCallback(data=val, encoder=encoder),
        early_stop_callback], **cfg.trainer)

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()
