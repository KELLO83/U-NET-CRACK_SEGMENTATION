import os
import logging
import argparse
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackseg.models import UNet
from crackseg.utils.dataset import RoadCrack
from crackseg.utils.general import random_seed
from crackseg.utils.losses import CrossEntropyLoss, DiceCELoss, DiceLoss, FocalLoss
import cv2
import numpy as np


@torch.inference_mode
def evaulte(model , data_loader,device):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    for image , target  in tqdm(data_loader , total=len(data_loader)):
        image , target = image.to(device) , target.to(device)
        with torch.no_grad():
            output = model(image)
            if model.out_channels == 1:
                output = F.sigmoid(output) > 0.5
            dice_loss = criterion(output,target)
            dice_score += 1 - dice_loss
            
    model.train()
    
    return dice_score / len(data_loader) , dice_loss


@torch.inference_mode
def image_segmentation_runnung(model , input , device):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    with torch.no_grad():
        image,mask = input[0].to(device) , input[1].to(device)
        output = model(image)
        dice_loss = criterion(output , mask)
        
    return output , dice_loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=2)
    weith_path = 'weights/best.pt'
    ckpt = torch.load(weith_path,map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict())
    model = model.to(device)
    
    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    
    test_dataset = RoadCrack(root=f'data/test',image_size=640,mask_suffix="",train=False)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=8,shuffle=False,pin_memory=True)
    
    for index , i in enumerate(test_loader):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        if index==3:
            batch = i
            image__ = batch[0]
            label__ = batch[1]
            break
    
    image = image__.squeeze().permute(1,2,0)
    image_np = image.detach().cpu().numpy()
    
    label_np = (label__ * 255).permute(1,2,0)
    label_np = label_np.detach().cpu().numpy()
    if label_np.ndim == 3 and label_np.shape[2] == 1:
        label_np = label_np[:,:,0].astype(np.uint8)
    
    image_np = np.flip(image_np , axis= -1)
    #label_np = np.flip(label_np  , axis= -1)
    
    ds , dl =evaulte(model,test_loader,device)
    print("============== Total Test set Dice socre : {:.5f} Dice loss : {:.5f}====================".format(ds,dl))
    
    for i in test_loader:
        batch = i
        break
    
    example = [image__ , label__]
    out , out_loss= image_segmentation_runnung(model , example , device)
    out = out.squeeze()
    result_np = out.detach().cpu().numpy()
    # print(out)
    # print(out.shape)
    print("Dice Score : {:.5f} Dice loss : {:.5f}".format(1-out_loss,out_loss))
    #cv2.imshow("Channel 1",result_np[0])
    cv2.imshow("CHannel 2",result_np[1])

    cv2.imshow("orgin image",image_np)
    cv2.imshow("label_np",label_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
    