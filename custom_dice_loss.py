import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDice_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(CustomDice_Loss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, input, target):
        pass
    

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
        
        #loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return dice_loss
