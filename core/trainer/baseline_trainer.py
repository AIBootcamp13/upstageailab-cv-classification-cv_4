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
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
import wandb

from core.models.convnext import ConvNeXt
from core.models.resnet50 import Resnet50
    
class BaselineModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if "convnext" in cfg.model.model.model_name:
            self.model = ConvNeXt(cfg)
        elif "resnet50" in cfg.model.model.model_name:
            self.model = Resnet50(cfg)

        self.criterion = nn.CrossEntropyLoss()

        n_classes = cfg.model.model.num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric = getattr(self, f"{stage}_f1")
        acc_metric.update(logits, y)
        f1_metric.update(logits, y)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def on_train_epoch_end(self):
        if self.current_epoch == self.cfg.freeze_epochs:
            print(f"Epoch {self.current_epoch+1}: Start Feature Extractor unfreeze and full-model fine-tuning")
            self.model.unfreeze()
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
        
        acc_metric.reset()
        f1_metric.reset()

    def configure_optimizers(self):
        opt = Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        if "scheduler" in self.cfg and self.cfg.scheduler is not None:
            sch = instantiate(self.cfg.scheduler, optimizer=opt)
            return {"optimizer": opt, "lr_scheduler": sch}
        return opt
    