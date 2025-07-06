import os
from typing import List
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning import Callback, Trainer, LightningModule
import numpy as np

class PerClassLossCallback(Callback):
    def __init__(self, num_classes: int, 
                 class_names: List[str] | None = None,
                 save_dir: str = "per_class_loss", every_n_epoch: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.class_names     = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.every_n_epoch = every_n_epoch
        self.save_dir = save_dir
        self.loss_sum = torch.zeros(num_classes)
        self.sample_sum = torch.zeros(num_classes)
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        logits = outputs["logits"]
        targets = outputs["targets"]
        
        loss_fn = pl_module.criterion

        targets = F.one_hot(targets, num_classes=self.num_classes).float()

        batch_loss = loss_fn(logits, targets, reduction="none").detach().cpu()

        probs      = targets.detach().cpu()                                  

        batch_loss = batch_loss.view(-1, 1)        
        self.loss_sum   += torch.sum(probs * batch_loss, dim=0)
        self.sample_sum += probs.sum(0)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epoch != 0:
            self._reset()
            return
        
        avg_loss = self.loss_sum / (self.sample_sum + 1e-8)
        avg_loss = avg_loss.numpy()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(self.num_classes), avg_loss, color="steelblue")
        
        if hasattr(self, "class_names"):
            ax.set_xticks(range(self.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticks(range(self.num_classes))
        
        ax.set_xlabel("Class")
        ax.set_ylabel("Avg Soft-Loss")
        ax.set_title(f"Per-Class Soft Loss @ epoch {epoch}")
        plt.tight_layout()
        os.makedirs(self.save_dir, exist_ok=True)
        fig.savefig(os.path.join(self.save_dir, f"soft_loss_epoch_{epoch:03d}.png"), dpi=150)
        plt.close(fig)
        self._reset()
    
    def _reset(self):
        self.loss_sum.zero_()
        self.sample_sum.zero_()