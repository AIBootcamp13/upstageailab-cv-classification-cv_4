import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None):
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data).values
        else:
            self.df = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target
    
class ImageDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.img_size = cfg.data.img_size
        self.data_path = cfg.data.data_path

        self.train_tf = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.val_tf = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            df = pd.read_csv(os.path.join(self.data_path, "train.csv"))
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            self.train_ds = ImageDataset(
                train_df, os.path.join(self.data_path, "train"), transform=self.train_tf
            )
            self.val_ds = ImageDataset(
                val_df, os.path.join(self.data_path, "train"), transform=self.val_tf
            )

        if stage in ("test", "predict", None):
            df = pd.read_csv(os.path.join(self.data_path, "sample_submission.csv"))
            self.test_ds = ImageDataset(
                df, os.path.join(self.data_path, "test"), transform=self.val_tf
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )