import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        print('initialize focalloss')
        super().__init__()
        self.gamma = gamma
        self.default_reduction = reduction.lower()

        # alpha 처리
        if isinstance(alpha, (DictConfig, ListConfig)):
            alpha = OmegaConf.to_container(alpha, resolve=True)
        if isinstance(alpha, list):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)

        
    def forward(self, logits, targets, reduction=None):
        reduction = (reduction or self.default_reduction).lower()
        B, C = logits.shape

        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()

        # ── hard label ↦ one-hot ─────────────────────────────
        if targets.dim() == 1:                       # hard label
            tgt_onehot = F.one_hot(targets, C).type_as(logits)
        else:                                        # 이미 soft label(one-hot/soft)
            tgt_onehot = targets.type_as(logits)

        ce = -(tgt_onehot * log_p).sum(dim=1)        # (B,)
        pt = (tgt_onehot * p).sum(dim=1)             # (B,)

        # alpha 가중치
        if self.alpha is not None:
            if targets.dim() == 1:
                alpha_t = self.alpha[targets.long()].to(logits.device)
            else:  # soft 라벨이면 가중 평균
                alpha_t = (tgt_onehot * self.alpha.to(logits.device)).sum(dim=1)
        else:
            alpha_t = 1.0

        loss = alpha_t * (1 - pt) ** self.gamma * ce  # focal

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
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