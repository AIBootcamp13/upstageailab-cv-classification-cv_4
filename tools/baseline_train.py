import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra

from core.datasets.baseline_dataset import ImageDataModule
from core.trainer.baseline_trainer import BaselineModule

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed if "seed" in cfg else 42, workers=True)

    data_module = ImageDataModule(cfg)
    model = BaselineModule(cfg)

    ckpt_cb = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k = 1,
        filename="best-{epoch:02d}-{val_f1:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        callbacks=[ckpt_cb, lr_monitor],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
