import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, smooth=1e-6):
    probs=torch.sigmoid(logits)
    probs=probs.view(-1)
    targets=targets.view(-1)
    inter=(probs*targets).sum()
    dice=(2*inter+smooth)/(probs.sum()+targets.sum()+smooth)
    return 1-dice

def combined_loss(logits, targets):
    return bce(logits, targets) + dice_loss(logits, targets)
