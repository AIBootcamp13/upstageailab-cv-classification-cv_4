import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import wandb
import torch
import numpy as np

from core.datasets.dataset import DatasetModule
from core.trainer.trainer import TrainerModule
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
            "model_name": cfg.model.model.model_name,
            "freeze_epochs": cfg.trainer.freeze_epochs,
            "batch_size": cfg.data.batch_size
        }
    )

    seed_everything(cfg.seed if "seed" in cfg else 42, workers=True)

    data_module = DatasetModule(cfg)
    data_module.setup('fit')

    samples = data_module.train_dataloader().dataset.samples
    labels = [label for _, label in samples]
    cls_counts = np.bincount(labels)
    total_cnt  = cls_counts.sum()
    alpha_np   = total_cnt / (len(cls_counts) * cls_counts)
    alpha_np   = alpha_np / alpha_np.sum()

    # ---------- 3. cfg 수정 (primitive 타입만!) ----------
    OmegaConf.set_struct(cfg, False)   # 구조 잠금 해제

    # loss_params는 DictConfig(dict) 여야 함
    cfg.loss.loss.gamma      = 2.0
    cfg.loss.loss.reduction  = "mean"
    cfg.loss.loss.alpha      = alpha_np.tolist()   # ← 리스트(float) OK

    # # scheduler primitive 값 저장
    # steps_per_epoch            = len(dm.train_dataloader())
    # cfg.scheduler.total_steps  = steps_per_epoch * cfg.trainer.max_epochs
    # cfg.scheduler.warmup_steps = steps_per_epoch * 3

    # cfg.loss_params["alpha"] = torch.tensor(total_count / (len(cls_counts) * cls_counts))
    # cfg.loss_params["gamma"] = 2.0
    # cfg.loss_params["reduction"] = "mean"
    
    cfg.scheduler.total_steps = len(data_module.train_dataloader()) * cfg.trainer.max_epochs
    cfg.scheduler.warmup_steps = len(data_module.train_dataloader()) * 3
    
    model = TrainerModule(cfg)

    ckpt_cb = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k = 1,
        filename="best-{epoch:02d}-{val_f1:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_cb = EarlyStopping(monitor=cfg.callback.monitor, mode=cfg.callback.mode, patience=cfg.callback.patience, min_delta=0.0005, verbose=True)
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cfg.get("bf16", False) else 32,
        callbacks=[ckpt_cb, lr_monitor, early_stop_cb],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()
