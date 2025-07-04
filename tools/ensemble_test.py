import os
import sys
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

import hydra
import pandas as pd
import numpy as np
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

def _find_fold_latest_ckpts(pattern: str = "lightning_logs/checkpoints/fold_*/best-*.ckpt") -> list[str]:
    return sorted(map(str, Path(".").glob(pattern)))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42), workers=True)

    ckpt_paths = _find_fold_latest_ckpts("lightning_logs/checkpoints/fold_*/best-*.ckpt")

    if not ckpt_paths:
        raise FileNotFoundError(
            "No checkpoint found. Train the model first or pass "
            "`test.ckpt_paths=[ckpt1,ckpt2,...]` via Hydra."
        )

    print(f"\n▶ Using checkpoints ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"  • {p}")

    dm = DatasetModule(cfg)

    fold_preds = []

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        logger=False, enable_checkpointing=False,
    )

    for path in ckpt_paths:
        if cfg.trainer.hnm.use_hnm == True:
            model = HardNegativeMiningTrainerModule.load_from_checkpoint(path, cfg=cfg)
        else:
            model = TrainerModule.load_from_checkpoint(path, cfg=cfg)
        model.eval()

        preds = trainer.predict(model, datamodule=dm)
        preds = torch.cat(preds).cpu().numpy()
        fold_preds.append(preds)

    ensemble_preds = np.mean(fold_preds, axis=0)

    sub_path = os.path.join(cfg.data.data_path, "sample_submission.csv")
    submission = pd.read_csv(sub_path)

    if ensemble_preds.ndim == 2:
        submission["target"] = ensemble_preds.argmax(axis=1)
    else:
        submission["target"] = ensemble_preds

    out_name = f"pred_softvote_{len(ckpt_paths)}f.csv"
    submission.to_csv(out_name, index=False)
    print(f"Soft-voting submission saved ➜ {out_name}")

if __name__ == "__main__":
    main()
