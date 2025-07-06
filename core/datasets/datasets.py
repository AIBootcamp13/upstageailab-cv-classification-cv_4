import os
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image

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

class AugraphyImageDataset(Dataset):
    def __init__(self, data, path, transform=None, is_test=False):
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data).values
        else:
            self.df = data.values
        self.path = path
        self.transform = transform

        self.samples = []
        self.is_test = is_test
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
        if self.is_test:
            if self.transform:
                img = self.transform(image=img)['image']
        else:
            if self.transform:
                if name.startswith("aug_"):
                    img = self.transform["Augraphy"](image=img)['image']
                else:
                    img = self.transform["NoAugraphy"](image=img)['image']
        return img, target