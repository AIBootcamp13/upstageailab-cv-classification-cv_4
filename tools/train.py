import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
import wandb

from core.datasets.baseline_dataset import ImageDataModule
from core.trainer.baseline_trainer import BaselineModule
from core.utils.utils import auto_increment_run_suffix

def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")

def get_latest_run(project_name, experiment_name):
    runs = get_runs(project_name)

    filtered = [
        run for run in runs
        if run.config.get("experiment_name") == experiment_name
    ]

    if not filtered:
        default_name = f"{experiment_name.replace('_', '-')}-000"
        return default_name
    
    return filtered[0].name

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = "CV4_Competition"
    try:
        run_name = get_latest_run(project_name, cfg.experiment_name)
    except Exception as e:
        print(f"[W&B WARNING] Failed to get previous runs: {e}")
        run_name = f"{cfg.experiment_name.replace('_', '-')}-000"
    next_run_name = auto_increment_run_suffix(run_name)
    wandb.init(
        project=project_name,
        id=next_run_name,
        notes="content-based classification model",
        tags=["content-based", "classification"],
        config={
            "experiment_name": cfg.experiment_name,
        }
    )

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
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        callbacks=[ckpt_cb, lr_monitor],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()
