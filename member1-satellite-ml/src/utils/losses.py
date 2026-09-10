import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self,logits,targets):
        bce=self.bce(logits,targets)
        probs=torch.sigmoid(logits)
        dims=(1,2,3)
        inter=(probs*targets).sum(dims)
        dice=(2*inter+1e-6)/(probs.sum(dims)+targets.sum(dims)+1e-6)
        return 0.5*bce+0.5*(1-dice.mean())
