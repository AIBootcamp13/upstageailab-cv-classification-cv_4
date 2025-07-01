import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from omegaconf import DictConfig
import hydra

class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
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

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret
    
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)

    # 시드를 고정합니다.
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data config
    data_path = './data'

    # model config
    model_name = cfg.model.model.model_name
    print(model_name)
    # training config
    img_size = 32
    LR = 1e-3
    EPOCHS = 1
    BATCH_SIZE = 32
    num_workers = 0

    # augmentation을 위한 transform 코드
    trn_transform = A.Compose([
        # 이미지 크기 조정
        A.Resize(height=img_size, width=img_size),
        # images normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
        ToTensorV2(),
    ])

    # test image 변환을 위한 transform 코드
    tst_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Dataset 정의
    trn_dataset = ImageDataset(
        os.path.join(data_path, "train.csv"),
        os.path.join(data_path, "train"),
        transform=trn_transform
    )
    tst_dataset = ImageDataset(
        os.path.join(data_path, "sample_submission.csv"),
        os.path.join(data_path, "test"),
        transform=tst_transform
    )
    print(len(trn_dataset), len(tst_dataset))

    # DataLoader 정의
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    tst_loader = DataLoader(
        tst_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
        ret['epoch'] = epoch

        log = ""
        for k, v in ret.items():
            log += f"{k}: {v:.4f}\n"
        print(log)

    preds_list = []

    model.eval()
    for image, _ in tqdm(tst_loader):
        image = image.to(device)

        with torch.no_grad():
            preds = model(image)
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = preds_list

    sample_submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    assert (sample_submission_df['ID'] == pred_df['ID']).all()

    pred_df.to_csv("pred.csv", index=False)

if __name__ == "__main__":
    main()