import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes Focal Loss.
        
        Args:
            inputs: Logits from the model (before sigmoid/softmax), shape [batch, num_classes] or [batch, C, H, W].
            targets: Ground truth labels (class indices), shape [batch] or [batch, H, W] for segmentation.
        
        Returns:
            Computed focal loss value.
        """

        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Binary Cross Entropy Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')

        # Probabilities after sigmoid
        probs = torch.sigmoid(inputs)
        p_t = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)

        # Focal Loss Computation
        F_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
