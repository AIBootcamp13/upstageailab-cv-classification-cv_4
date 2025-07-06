import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── ArcMarginProduct (ArcFace 헤드) ───────────────────────────
class ArcMarginProduct(nn.Module):
    """
    in_features → cosine margin 적용 → scaled logits
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)

    def forward(self, x, labels):
        # cosine θ
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine   = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        one_hot = F.one_hot(labels, cosine.size(1)).float().to(x.device)
        logits  = self.s * (one_hot * phi + (1 - one_hot) * cosine)
        return logits


# ── ConvNeXt backbone + ArcMarginProduct ─────────────────────
class ConvNeXtArcFace(nn.Module):
    def __init__(self, cfg, s=30.0, m=0.50):
        super().__init__()
        print("create model ConvNeXt + ArcFace")
        
        # timm 백본, 분류 헤드 제거 (num_classes=0)
        self.backbone = timm.create_model(
            cfg.model.model.model_name,
            pretrained=cfg.model.model.pretrained,
            num_classes=0
        )
        in_feats = self.backbone.num_features
        n_classes = cfg.model.model.num_classes

        # ArcMarginProduct head
        self.arc_head = ArcMarginProduct(in_feats, n_classes, s=s, m=m)

        # ── (선택) Freeze strategy ───────────────────────────
        for name, param in self.backbone.named_parameters():
            if cfg.trainer.freeze_epochs > 0:  # 초기 epochs만 freeze
                if 'head' in name or 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True  # 바로 fine-tune 전체

    def forward(self, x, labels=None):
        feats  = self.backbone(x) 
        if labels is None:
            logits = F.linear(F.normalize(feats), F.normalize(self.arc_head.weight)) * self.arc_head.s
        else:
            logits = self.arc_head(feats, labels)

        return logits

    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True