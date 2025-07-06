import os

import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Callback, Trainer, LightningModule
from collections import defaultdict
import numpy as np

class PerClassLossCallback(Callback):
    def __init__(self, num_classes: int, save_dir: str = "per_class_loss", every_n_epoch: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.every_n_epoch = every_n_epoch
        self.save_dir = save_dir
        self.loss_sum = torch.zeros(num_classes)
        self.sample_sum = torch.zeros(num_classes)
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        logits = outputs["logits"]
        targets = outputs["targets"]
        loss_fn = pl_module.criterion

        batch_loss = loss_fn(logits, targets, reduction="none").detach().cpu()
        probs      = targets.detach().cpu()

        self.loss_sum   += torch.sum(probs.T * batch_loss, dim=1)
        self.sample_sum += probs.sum(0)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epoch != 0:
            self._reset()
            return
        
        avg_loss = self.loss_sum / (self.sample_sum + 1e-8)
        avg_loss = avg_loss.numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(self.num_classes), avg_loss, color="steelblue")
        ax.set_xlabel("Class"); ax.set_ylabel("Avg Soft-Loss")
        ax.set_title(f"Per-Class Soft Loss @ epoch {epoch}")
        plt.tight_layout()
        os.makedirs(self.save_dir, exist_ok=True) 
        fig.savefig(os.path.join(self.save_dir, f"soft_loss_epoch_{epoch:03d}.png"), dpi=150)
        plt.close(fig)
        self._reset()
    
    def _reset(self):
        self.loss_sum.zero_()
        self.sample_sum.zero_()