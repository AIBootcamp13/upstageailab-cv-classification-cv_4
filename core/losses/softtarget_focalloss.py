import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig

class SoftTargetFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        print('initialize SoftTargetFocalLoss')
        self.alpha = alpha
        self.gamma = gamma
        self.default_reduction = reduction

        if self.alpha is not None:
            if isinstance(self.alpha, (DictConfig, ListConfig)):
                self.alpha = OmegaConf.to_container(self.alpha, resolve=True)
            if isinstance(self.alpha, list):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)

    def forward(self, logits, targets, reduction=None):
        reduction = reduction or self.default_reduction

        alpha = self.alpha.to(targets.device)
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()        

        ce_per_cls = -targets * log_p
        focal_per  = (1 - p).pow(self.gamma)
        if alpha is not None:
            alpha = alpha.to(logits.device)
            fl = alpha * ce_per_cls * focal_per
        else:
            fl = ce_per_cls * focal_per

        loss = fl.sum(dim=1)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:                                 
            return loss