import torch
import torch.nn.functional as F


class CustomDice_Loss(torch.nn.Module):
    def __init__(self , epsilon=1e-7):
        super(CustomDice_Loss , self).__init__()
        self.epsilon = epsilon
        
        
    def forward(self,input,target):
        logits = torch.sigmoid(input)
        
        input =  input.flatten()
        target = target.flatten()
        
        intersection = (input * target).sum()
        
        dice_score = (2 * intersection + self.epsilon) / (input.sum() + target.sum() + self.epsilon)
        
        dice_loss = 1 - dice_score
        
        return dice_loss
    
    
    
    
    