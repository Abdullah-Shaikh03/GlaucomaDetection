import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert target indices to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Calculate binary cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        
        # Calculate focal loss components
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss