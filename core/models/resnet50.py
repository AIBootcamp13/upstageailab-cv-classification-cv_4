import torch
import torch.nn as nn
import timm

class Resnet50(nn.Module):
    def __init__(self, cfg):
        super(Resnet50, self).__init__()
        print("create model resnet50")
        self.backbone = timm.create_model(cfg.model.model.model_name,
                                          pretrained=cfg.model.model.pretrained,
                                          num_classes=cfg.model.model.num_classes)
        
        for name, param in self.backbone.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def unfreeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True