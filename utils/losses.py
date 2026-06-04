import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Focal Loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        focal_loss = focal_loss.mean()

        # Dice Loss
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return focal_loss + dice_loss

class JointMultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha     # Segmentation weight
        self.beta = beta       # Classification weight
        self.gamma = gamma     # Multi-scale MSE weight
        
        self.seg_criterion = DiceFocalLoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()

    def forward(self, cls_out, seg_out, ds_outs, cls_target, seg_target):
        # 1. Classification Loss
        loss_cls = self.cls_criterion(cls_out, cls_target)
        
        # 2. Main Segmentation Loss
        loss_seg = self.seg_criterion(seg_out, seg_target)
        
        # 3. Multi-Scale MSE Anchoring Loss (Eq 3.14)
        loss_ms = 0
        for ds in ds_outs:
            # Resize ground truth to match intermediate decoder resolutions
            target_resized = F.interpolate(seg_target, size=ds.shape[2:], mode='nearest')
            loss_ms += self.mse_criterion(torch.sigmoid(ds), target_resized)
        loss_ms = loss_ms / len(ds_outs)
        
        # Combined Loss (Eq 3.7)
        total_loss = (self.alpha * loss_seg) + (self.beta * loss_cls) + (self.gamma * loss_ms)
        
        return total_loss, loss_seg, loss_cls, loss_ms