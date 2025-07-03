import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from albumentations.pytorch import ToTensorV2
from PIL import Image
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F  
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
import wandb
from timm.data.mixup import Mixup

from core.models.convnext import ConvNeXt
from core.models.resnet50 import Resnet50
from core.models.vit import ViT
from core.models.swinTransformer import SwinTransformer
from core.models.efficientnet import EfficientNet
from core.losses.focalloss import FocalLoss
from core.losses.softtarget_focalloss import SoftTargetFocalLoss
from core.callbacks.ema import EMA

class TrainerModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if "convnext" in cfg.model.model.model_name:
            self.model = ConvNeXt(cfg)
        elif "resnet50" in cfg.model.model.model_name:
            self.model = Resnet50(cfg)
        elif "deit" in cfg.model.model.model_name:
            self.model = ViT(cfg)
        elif "swin" in cfg.model.model.model_name:
            self.model = SwinTransformer(cfg)
        elif "efficientnet" in cfg.model.model.model_name:
            self.model = EfficientNet(cfg)
        
        if cfg.loss.loss_name == "focalloss":
            self.criterion = FocalLoss(**cfg.loss.loss)
        elif cfg.loss.loss_name == "softtarget_focalloss":
            self.criterion = SoftTargetFocalLoss(**cfg.loss.loss)

        # self.criterion = nn.CrossEntropyLoss()

        n_classes = cfg.model.model.num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")

        self.ema = None

        if cfg.trainer.use_mixup == True:
            self.mixup = Mixup(
                mixup_alpha=0.4,
                cutmix_alpha=0.0,
                prob=1.0,
                switch_prob= 0.0,
                mode="batch",
                label_smoothing=0.0,
                num_classes=n_classes
            )
        else:
            self.mixup = None

    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        if self.cfg.trainer.use_ema == True:
            self.ema = EMA(self.model, decay=0.995)
            if hasattr(self.ema, "ema_model"):
                self.ema.ema_model.to(self.device)
                self.ema.ema_model.eval()

    def on_predict_start(self):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            self.ema.ema_model.to(self.device)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch

        if stage == "train":
            if self.mixup is not None:
                x, y = self.mixup(x, y)

        logits = self(x)
        loss = self.criterion(logits, y)

        y_hard = y.argmax(1) if y.ndim == 2 else y

        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        acc_metric.update(logits, y_hard)
        f1_metric.update(logits, y_hard)
    
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            if self.current_epoch >= self.cfg.trainer.ema_update_epochs:
                self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict())
            loss = self._shared_step(batch, "val")
            self.model.load_state_dict(backup_state)
            return loss
        
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            logits = self.ema.ema_model(x)
        else:
            logits = self(x)
        return F.softmax(logits, dim=1)

    def on_train_epoch_end(self):
        if self.current_epoch == self.cfg.trainer.freeze_epochs:
            print(f"Epoch {self.current_epoch+1}: Start Feature Extractor unfreeze and full-model fine-tuning")
            self.model.unfreeze()

            # optimizer = self.trainer.optimizers[0]
            # for pg in optimizer.param_groups:
            #     old_lr = pg["lr"]
            #     pg["lr"] = old_lr * 0.1
            #     print(f"LR {old_lr:.6f} → {pg['lr']:.6f}")

            # scheduler = self.lr_schedulers()
            # if hasattr(scheduler, "base_lrs"):
            #     scheduler.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def _log_epoch_metrics(self, stage: str):
        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)

        wandb.log({f"Accuracy/{stage}": acc})
        wandb.log({f"F1score/{stage}": f1})
        
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        wandb.log({f"LR/{stage}": current_lr})

        acc_metric.reset()
        f1_metric.reset()

    def on_save_checkpoint(self, checkpoint):
        if self.cfg.trainer.use_ema == True and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("ema_state") and self.cfg.trainer.use_ema == True:
            decay = float(getattr(self.cfg.trainer, "ema_decay", 0.995))
            self.ema = EMA(self.model, decay=decay)
            self.ema.ema_model.load_state_dict(checkpoint["ema_state"])
            self.ema.ema_model.eval()

    def configure_optimizers(self):
        optimizer_name = str(self.cfg.optimizer._target_)
        print(f"=========== {optimizer_name} ==============")
        if "AdamW" in optimizer_name:
            print("======== AdamW =========")
            opt = AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

            scheduler = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.cfg.scheduler.warmup_steps,
                num_training_steps=self.cfg.scheduler.total_steps
            )
            return {
                "optimizer":   opt,
                "lr_scheduler": {
                    "scheduler":  scheduler,
                    "interval":   "step",   # ← 매 step마다 step()
                    "frequency":  1,
                    "name":       "cosine_warmup",
                },
            }
        else:
            opt = Adam(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )


        return opt