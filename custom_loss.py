import torch
import torch.nn as nn
import torch.nn.functional as F



class CustomCrossEntropy(nn.Module):
    def __init__(self):
        super(CustomCrossEntropy, self).__init__()
        self.epsilon = 1e-12
        
    def forward(self, inputs, target):
        # Apply sigmoid to the inputs to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Clamp values to avoid log(0)
        inputs = torch.clamp(inputs, self.epsilon, 1 - self.epsilon)

        # Compute binary cross entropy loss
        loss = -(target * torch.log(inputs) + (1 - target) * torch.log(1 - inputs))
        
        # Return mean loss
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-7, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        
    def forward(self, input, target):
        # Sigmoid를 적용하여 확률값으로 변환
        input = torch.sigmoid(input)
        
        # BCE Loss 계산
        bce_loss = self.bce_loss(input, target.float())
        
        # Dice Loss 계산
        intersection = (input * target).sum(dim=(2, 3))
        dice_score = (2 * intersection + self.epsilon) / (input.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.epsilon)
        dice_loss = 1 - dice_score.mean()
        
        # BCE와 Dice Loss를 결합
        
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: model outputs (B, C, H, W) - in this case, (4, 1, 512, 512)
        # targets: ground truth labels (B, H, W) - in this case, (4, 512, 512)
     
        # Compute the cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute the probability of the class
        probs = torch.sigmoid(inputs)
        probs = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute the focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Compute the final focal loss
        focal_loss = self.alpha * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss