import os
import logging
import argparse
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackseg.models import UNet
from crackseg.utils.dataset import RoadCrack,CustomDataset
from crackseg.utils.general import random_seed
from crackseg.utils.losses import CrossEntropyLoss, DiceCELoss, DiceLoss, FocalLoss
import cv2
import numpy as np

from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision , BinaryRecall
from torchmetrics.classification import BinaryJaccardIndex
import torchmetrics

from sklearn.metrics import average_precision_score , precision_recall_curve

def get_ap(output,target):
    output = torch.sigmoid(output)
    
    output = output.contiguous().flatten().detach().cpu().numpy()
    target = target.contiguous().flatten().detach().cpu().numpy()
    
    ap_score = average_precision_score(target , output)

    print("Ap score : {:.5f}%".format(ap_score*100))



def get_metrics(output,target,th=0.01):
    output = torch.sigmoid(output)
    
    output = (output > th).float()
    
    TP = (output * target).sum().item()
    FP = (output * (1-target)).sum().item()
    FN = ((1-output) * target).sum().item()
    
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    print("정밀도 : {:.5f}% 재현율 : {:.5f}% f1_score : {:.5f}%".format(precision*100,recall*100,f1_score*100))

@torch.inference_mode
def evaulte(model , data_loader,device):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    for image , target  in tqdm(data_loader , total=len(data_loader)):
        image , target = image.to(device) , target.to(device)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output,target)
            
    model.train()
    
    return output , loss

@torch.inference_mode
def image_segmentation_runnung(model , input , device,th):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        image,mask = input[0].to(device) , input[1].to(device)
        output = model(image)
        loss = criterion(output , mask)
        get_ap(output,mask)
        get_metrics(output,mask,th)
        
    return output , loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=1)
    weith_path = 'weights/CRKWH100/last.pt'
    ckpt = torch.load(weith_path,map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict())
    model = model.to(device)
    
    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    
    image_path = 'data/CRKWH100_IMAGE/val'
    mask_path = 'data/CRKWH100_MASK/val'
    test_dataset = CustomDataset(image_path,mask_path,'png',is_resize=True)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=8,shuffle=False,pin_memory=True)
    
    for index , i in enumerate(test_loader):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        if index==2:
            t = i
            image__ = t[0]
            label__ = t[1]
            break
    
    out , loss = evaulte(model,test_loader,device)
    print("Test Dataset loss :{:.5f}".format(loss))
    
    target =  image__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    mask = label__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    target = (target * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)

    threshold = 1e-1
    
    example = [image__ , label__]
    out_image , out_loss= image_segmentation_runnung(model , example , device , threshold)
    out_image = torch.sigmoid(out_image)
    out_image = (out_image > threshold).float() * 255
    out_image = out_image.to(torch.uint8)
    out_image = out_image.squeeze(0).permute(1,2,0).detach().cpu().numpy()

    
    win_image_name = 'image'
    mask_image_name = 'mask'
    out_image_name = 'out'
    
    cv2.namedWindow(win_image_name)
    cv2.namedWindow(mask_image_name)
    cv2.namedWindow(out_image_name)
    
    cv2.moveWindow(win_image_name,1200,1200)
    cv2.moveWindow(mask_image_name,1700,1200)
    cv2.moveWindow(out_image_name , 2200 , 1200)
    
    cv2.imshow(win_image_name,target)
    cv2.imshow(mask_image_name,mask)
    cv2.imshow(out_image_name,out_image)
    
    cv2.waitKey()
    cv2.destroyAllWindows()