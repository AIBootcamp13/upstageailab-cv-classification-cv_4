import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate,
    ColorJitter, RandomBrightnessContrast, CLAHE,
    GaussianBlur, CoarseDropout, Resize, Normalize
)
from PIL import Image
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, data, path, transform=None, is_test=False):
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data).values
        else:
            self.df = data.values
        self.path = path
        self.transform = transform

        self.samples = []
        if is_test:
            pass
        else:
            for name, target in self.df:
                self.samples.append((name, target))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target
    
class DatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.img_size = cfg.data.img_size
        self.data_path = cfg.data.data_path

        self.train_tf = Compose([
            # 크롭/회전/플립
            RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.8, 1.0), p=1.0),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.3),
            Rotate(limit=15, p=0.5),

            # 색상 변화 (RandAugment 대체 조합)
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # RandomContrast(limit=0.2, p=0.3),
            CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

            # 블러 / 지우기
            GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.3),
            CoarseDropout(
                max_holes=1,
                max_height=int(self.img_size * 0.2),
                max_width=int(self.img_size * 0.2),
                fill_value=0,
                p=0.25
            ),

            # 정규화 + Tensor
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        # --- Validation / Test --------------------------------------------------
        self.val_tf = Compose(
            [
                Resize(self.img_size, self.img_size),
                Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.train_idx = None
        self.val_idx = None
        self.train_df = None
        self.val_df = None
    
    def set_split_idx(self, train_idx, val_idx):
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            self.full_df = pd.read_csv(os.path.join(self.data_path, "train.csv"))

            if self.train_idx is not None and self.val_idx is not None:
                self.train_df = self.full_df.iloc[self.train_idx].reset_index(drop=True)
                self.val_df = self.full_df.iloc[self.val_idx].reset_index(drop=True)
            else:
                targets = self.full_df["target"].tolist()
                self.train_df, self.val_df = train_test_split(
                    self.full_df, test_size=0.2, stratify=targets, random_state=42
                )

            self.train_ds = ImageDataset(
                self.train_df, os.path.join(self.data_path, "train"), transform=self.train_tf
            )
            self.val_ds = ImageDataset(
                self.val_df, os.path.join(self.data_path, "train"), transform=self.val_tf
            )

        if stage in ("test", "predict", None):
            df = pd.read_csv(os.path.join(self.data_path, "sample_submission.csv"))
            self.test_ds = ImageDataset(
                df, os.path.join(self.data_path, "test"), transform=self.val_tf
            )

    def set_train_dataset(self, new_df):
        self.train_ds = ImageDataset(
            new_df, 
            os.path.join(self.data_path, "train"),
            transform=self.train_tf,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
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