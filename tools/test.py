import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from core.datasets.dataset import DatasetModule
from core.trainer.trainer import TrainerModule
from core.trainer.HNMTrainer import HardNegativeMiningTrainerModule


def _find_latest_ckpt() -> str | None:
    from pathlib import Path
    ckpts = list(Path(".").rglob("best-*.ckpt"))
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]) if ckpts else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42), workers=True)

    ckpt_path: str | None = cfg.get("test", {}).get("ckpt_path", None)  # hydra override 가능
    ckpt_path = ckpt_path or _find_latest_ckpt()
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint (.ckpt) found. Train the model first or specify test.ckpt_path=<path>.")

    print(f"Using checkpoint: {ckpt_path}")

    dm = DatasetModule(cfg)
    if cfg.trainer.hnm.use_hnm == True:
        model = HardNegativeMiningTrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        model = TrainerModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()

    trainer = Trainer(accelerator="auto", devices="auto", precision="bf16-mixed" if cfg.get("bf16", False) else 32)
    preds = trainer.predict(model, datamodule=dm)       
    preds = torch.cat(preds, dim=0)                     
    preds = torch.argmax(preds, dim=1).cpu().numpy()    

    submission = pd.read_csv(os.path.join(cfg.data.data_path, "sample_submission.csv"))
    submission["target"] = preds
    submission.to_csv("pred.csv", index=False)


if __name__ == "__main__":
    main()
