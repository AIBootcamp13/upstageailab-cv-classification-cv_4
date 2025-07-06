import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pytorch_lightning import LightningModule, Trainer, Callback

class TSNECallback(Callback):
    def __init__(self, save_dir="error_tsne", every_n_epoch: int = 5, sample_limit: int = 1500):
        super().__init__()
        self.every_n_epoch = every_n_epoch
        self.sample_limit = sample_limit
        self.save_dir = save_dir
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epoch != 0:
            return
        
        pl_module.eval()
        feats, labels = [], []
        collected = 0

        val_loader = trainer.datamodule.val_dataloader()
        for x, y in val_loader:
            x = x.to(pl_module.device, non_blocking=True)

            f = pl_module.backbone.forward_features(x) 

            feats.append(f.cpu())
            labels.append(y.cpu())

            collected += f.size(0)
            if collected >= self.sample_limit:
                break
        
        feats = torch.cat(feats)[: self.sample_limit].numpy()
        labels = torch.cat(labels)[: self.sample_limit].numpy()

        feats_50 = PCA(n_components=50, random_state=42).fit_transform(feats)
        emb = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42).fit_transform(feats_50)

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=labels, s=6, cmap="tab20", alpha=0.85)
        ax.set_title(f"t-SNE (val) @ epoch {epoch}")
        ax.axis("off")

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig_path = os.path.join(self.save_dir, f"tsne_epoch_{epoch:03d}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")

        plt.close(fig)