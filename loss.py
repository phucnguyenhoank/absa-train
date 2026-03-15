import torch
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2, alpha=None):
    ce_losses = F.cross_entropy(logits, targets, reduction="none")

    pt = torch.exp(-ce_losses)

    focal = (1 - pt) ** gamma * ce_losses

    if alpha is not None:
        focal = alpha[targets] * focal

    return focal.mean()
