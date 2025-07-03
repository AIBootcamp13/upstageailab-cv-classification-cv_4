import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        print('initialize focalloss')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
        if self.alpha is not None:
            if isinstance(self.alpha, (DictConfig, ListConfig)):
                self.alpha = OmegaConf.to_container(self.alpha, resolve=True)
            if isinstance(self.alpha, list):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
    def forward(self, logits, targets):
        alpha = self.alpha.to(targets.device)
        # self.gamma = self.gamma.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)              

        if alpha is not None:
            at = alpha[targets]           
        else:
            at = 1.0

        fl = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        else:                                 
            return fl
        # print(targets.device)
        # ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # pt = torch.exp(-ce_loss)
        # at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        # fl = at * (1 - pt) ** self.gamma * ce_loss
        # if self.reduction == 'mean':
        #     return fl.mean()
        # elif self.reduction == 'sum':
        #     return fl.sum()
        # else:
        #     return fl