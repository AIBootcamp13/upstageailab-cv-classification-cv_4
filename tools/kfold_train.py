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
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.datasets.dataset import DatasetModule
from core.trainer.trainer import TrainerModule
from core.trainer.HNMTrainer import HardNegativeMiningTrainerModule
from core.callbacks.hardNegativeMining import HNMCallback
from core.callbacks.errorAnalysis import ErrorAnalysisCallback
from core.callbacks.tsne import TSNECallback
from core.callbacks.perClassLoss import PerClassLossCallback
from core.callbacks.confusionMatrix import ConfusionMatrixCallback
from core.utils.utils import auto_increment_run_suffix, get_latest_run, project_path, make_error_run_dir

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = "CV4_Competition_KFold"
    # try:
    #     run_name = get_latest_run(project_name, cfg.experiment_name)
    # except Exception as e:
    #     print(f"[W&B WARNING] Failed to get previous runs: {e}")
    #     run_name = f"{cfg.experiment_name.replace('_', '-')}-000"
    # next_run_name = auto_increment_run_suffix(run_name)
    # wandb.init(
    #     project=project_name,
    #     id=next_run_name,
    #     notes="content-based classification model",
    #     tags=["content-based", "classification"],
    #     config={
    #         "experiment_name": cfg.experiment_name,
    #         "model_name": cfg.model.model.model_name,
    #         "freeze_epochs": cfg.trainer.freeze_epochs,
    #         "batch_size": cfg.data.batch_size
    #     }
    # )
    OmegaConf.set_struct(cfg, False)
    
    seed_everything(cfg.seed if "seed" in cfg else 42, workers=True)

    data_module = DatasetModule(cfg)
    origin_dataset = data_module.origin_dataset
    origin_labels = [y for _, y in origin_dataset]

    kfold = StratifiedKFold(
        n_splits = 5,
        shuffle=True,
        random_state=42
    )

    error_root_dir = make_error_run_dir()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(origin_labels)), origin_labels)):
        # run_name = f"{cfg.experiment_name}-fold{fold}"
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
                "freeze_epochs": cfg.model.model.freeze_epochs,
                "batch_size": cfg.data.batch_size
            }
        )

        data_module.set_split_idx(train_idx, val_idx)
        data_module.setup("fit")

        samples = data_module.train_dataloader().dataset.samples
        labels = [label for _, label in samples]
        cls_counts = np.bincount(labels)
        total_cnt  = cls_counts.sum()
        alpha_np   = total_cnt / (len(cls_counts) * cls_counts)
        alpha_np   = alpha_np / alpha_np.sum()

        cfg.loss.loss.gamma      = 2.0
        cfg.loss.loss.reduction  = "mean"
        cfg.loss.loss.alpha      = alpha_np.tolist()
        
        cfg.scheduler.total_steps = len(data_module.train_dataloader()) * cfg.trainer.max_epochs
        cfg.scheduler.warmup_steps = len(data_module.train_dataloader()) * 3

        if cfg.trainer.hnm.use_hnm == True:
            model = HardNegativeMiningTrainerModule(cfg)
        else:
            model = TrainerModule(cfg)

        ckpt_cb = ModelCheckpoint(
            dirpath=f"lightning_logs/checkpoints/fold_{fold}",
            monitor="val_f1",
            mode="max",
            filename=f"best-f{fold}-" + "{epoch:02d}-{val_f1:.3f}",
            save_top_k=1,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        early_stop_cb = EarlyStopping(monitor=cfg.callback.monitor, mode=cfg.callback.mode, patience=cfg.callback.patience, min_delta=0.0005, verbose=True)

        error_cb = ErrorAnalysisCallback(num_classes=cfg.model.model.num_classes, 
                                         class_names=data_module.meta_df["class_name"].unique(), 
                                         save_dir=os.path.join(project_path(), error_root_dir, f"{cfg.trainer.error.analysis.save_dir}_fold_{fold}"), 
                                         top_k=cfg.trainer.error.analysis.top_k)

        tsne_cb = TSNECallback(num_classes=cfg.model.model.num_classes, 
                            class_names=data_module.meta_df["class_name"].unique(), 
                            save_dir=os.path.join(project_path(), error_root_dir, f"{cfg.trainer.error.tsne.save_dir}_fold_{fold}"), 
                            every_n_epoch=cfg.trainer.error.tsne.every_n_epoch)

        pcl_cb = PerClassLossCallback(num_classes=cfg.model.model.num_classes, 
                                    class_names=data_module.meta_df["class_name"].unique(), 
                                    save_dir=os.path.join(project_path(), error_root_dir, f"{cfg.trainer.error.perclassloss.save_dir}_fold_{fold}"))

        cm_cb = ConfusionMatrixCallback(num_classes=cfg.model.model.num_classes, 
                                        class_names=data_module.meta_df["class_name"].unique(), 
                                        save_dir=os.path.join(project_path(), error_root_dir, f"{cfg.trainer.error.confusion_matrix.save_dir}_fold_{fold}"))
    
        if cfg.trainer.hnm.use_hnm == True:
            hnm_cb = HNMCallback(data_module.train_df, train_idx=data_module.train_idx, cfg=cfg)
            callbacks = [ckpt_cb, lr_monitor, early_stop_cb, hnm_cb, error_cb, tsne_cb, pcl_cb, cm_cb]
        else:
            callbacks = [ckpt_cb, lr_monitor, early_stop_cb, error_cb, tsne_cb, pcl_cb, cm_cb]

        trainer = Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator="auto",
            devices="auto",
            precision="bf16-mixed" if cfg.get("bf16", False) else 32,
            callbacks=callbacks,
            log_every_n_steps=1,
            reload_dataloaders_every_n_epochs=1)

        trainer.fit(model, datamodule=data_module)

        wandb.finish()

if __name__ == "__main__":
    main()
