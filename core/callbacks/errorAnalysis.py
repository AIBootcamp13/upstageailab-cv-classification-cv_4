import os

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback

class ErrorAnalysisCallback(Callback):
    def __init__(self, save_dir="error_analysis", top_k=10):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.top_k = top_k
        self.reset_buffer()

    def reset_buffer(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []
        self.all_fnames = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = pl_module(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if hasattr(batch, 'filenames'):
            fnames = batch.filenames
        elif hasattr(pl_module.trainer.datamodule.val_ds, "samples"):
            fnames = [pl_module.trainer.datamodule.val_ds.samples[idx][0] for idx in range(batch_idx * len(x), (batch_idx + 1) * len(x))]
        else:
            fnames = [f"sample_{batch_idx}_{i}.png" for i in range(len(x))]

        self.all_preds.extend(preds.cpu().numpy())
        self.all_probs.extend(probs.max(dim=1).values.cpu().numpy())
        self.all_targets.extend(y.cpu().numpy())
        self.all_fnames.extend(fnames)
    

    def on_validation_epoch_end(self, trainer, pl_module):
        df = pd.DataFrame({
            "filename": self.all_fnames,
            "target": self.all_targets,
            "pred": self.all_preds,
            "confidence": self.all_probs
        })

        error_df = df[df["target"] != df["pred"]]
        error_df = error_df.sort_values("confidence", ascending=False)

        save_df = error_df.head(self.top_k)
        save_path = os.path.join(self.save_dir, f"val_errors_epoch{trainer.current_epoch:03d}.csv")
        save_df.to_csv(save_path, index=False)
        print(f"[ErrorAnalysis] Saved {len(save_df)} misclassified samples → {save_path}")

        self.reset_buffer()