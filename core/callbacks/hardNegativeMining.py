import pandas as pd
from pytorch_lightning import Callback

from core.datasets.dataset import ImageDataset

class HardNegativeMiningCallback(Callback):
    def __init__(self, base_df: pd.DataFrame, train_idx: list[int], class_aug_cnt: dict[int, int]):
        self.baes_df = base_df
        self.train_idx = train_idx
        self.class_aug_cnt = class_aug_cnt
    
    def on_train_epoch_end(self, trainer, pl_module):
        per_cls_loss = pl_module.compute_per_class_loss()
        epoch = trainer.current_epoch

        new_df = pl_module.prepare_train_df_for_epoch(
            base_df = self.base_df,
            per_cls_loss = per_cls_loss,
            epoch = epoch,
            class_aug_cnt = self.class_aug_cnt,
            train_idx = self.train_idx,
        )
        if new_df is self.base_df:
            return
        
        new_dataset = ImageDataset(new_df, transform=pl_module.train_df)
        trainer.train_dataloader().dataset.dataset = new_dataset

        print(f"▶ Train set size → {len(new_dataset)}")