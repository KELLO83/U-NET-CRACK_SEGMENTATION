import os
import logging
import argparse
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crackseg.models.resuidual_unet import ResUNet
from crackseg.models import UNet
from crackseg.utils.dataset import RoadCrack,CustomDataset
from crackseg.utils.general import random_seed
from custom_loss import CustomCrossEntropy , BCEDiceLoss , FocalLoss
import cv2
import numpy as np

from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision , BinaryRecall
from torchmetrics.classification import BinaryJaccardIndex
from sklearn.metrics import average_precision_score


def get_ap(output,target):
    output = torch.sigmoid(output)
    
    output = output.contiguous().flatten().detach().cpu().numpy()
    target = target.contiguous().flatten().detach().cpu().numpy()
    
    ap_score = average_precision_score(target , output)

    print("Ap score : {:.5f}%".format(ap_score*100))


def find_best_threshold(outputs, targets):
    thresholds = np.arange(0, 1.01, 0.01)
    best_f1 = 0.0
    best_thresholds = 0.0
    for i in thresholds:
        f1 = get_metrics(outputs,targets,i)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = i
            print("best f1 update : {:.4f} best th update : {:.4f}".format(best_f1,best_thresholds))
    
    print("best thresholds : ",best_thresholds)

    return best_thresholds

 
def get_metrics(out_list,th=0.5,flag=False):
    
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    for output,target in out_list:
        output = torch.sigmoid(output)
        
        output = (output > th).float()
        
        TP = (output * target).sum().item()
        FP = (output * (1-target)).sum().item()
        FN = ((1-output) * target).sum().item()
        
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        p = sum(precision_list) / len(precision_list)
        r = sum(recall_list) / len(recall_list)
        f = sum(f1_score) / len(f1_score)
        
    print (f"임계점 : {th:.5f } ------> 정밀도평균  : {p:.5f} 재현율평균 : {r:.5f} f1 score평균 : {f:.5f}")
    
    return th , f


@torch.inference_mode
def evaulte(model , data_loader,device):
    model.eval()
    loss_list = []
    out_list = []
    criterion = BCEDiceLoss()
    for image , target  in tqdm(data_loader , total=len(data_loader)):
        image , target = image.to(device) , target.to(device)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output,target)
            out_list.append(output,target)
            loss_list.append(loss.item())
    
    it_range = np.arange(0.1 , 1.01 , 0.1)
    best_th = 0.0
    best_f = 0.0
    for th in it_range:
        th_res , f = get_metrics(out_list)   
        if f > best_f:
            print(f"Threshold :  {th_res:.2f} update best f1 score ------> ",f)
            best_f = f 
            best_th = th_res
                     
    return sum(loss_list) / len(loss_list)


@torch.inference_mode
def image_segmentation_runnung(model , input , device):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        image,mask = input[0].to(device) , input[1].to(device)
        output = model(image)
        loss = criterion(output , mask)
        th  = find_best_threshold(output , mask)
        get_ap(output,mask)
        get_metrics(output,mask,th,flag=True)
        
    return output , loss , th


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = ResUNet(in_channels=3, out_channels=1)
    weight_path = 'weights/CRKWH100/dice_only/best.pt'
    #weight_path = 'weights/CrackLS315/custom_cross/best.pt'
    ckpt = torch.load(weight_path,map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict())
    model = model.to(device)
    
    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    

    image_path = 'data/CRKWH100_IMAGE/test'
    mask_path = 'data/CRKWH100_MASK/test'
    image_type = 'png'
    test_dataset = CustomDataset(image_path,mask_path,image_type,is_resize=True)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=8,shuffle=False,pin_memory=True)
    
    for index , i in enumerate(test_loader):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        if index == 4:
            t = i
            image__ = t[0]
            label__ = t[1]
            break
    
    loss = evaulte(model,test_loader,device)
    print("Test Dataset loss :{:.5f}".format(loss))
    
    target =  image__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    mask = label__.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    target = (target * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)

    
    example = [image__ , label__]
    out_image , out_loss , th =  image_segmentation_runnung(model , example , device )
    out_image = torch.sigmoid(out_image)
    print(f"out image loss : {out_loss:.3f}")
    print(f"Image range min : {torch.min(out_image):.5f} max : {torch.max(out_image):.5f}")
    out_image = (out_image > th).float() * 255
    out_image = out_image.to(torch.uint8)
    out_image = out_image.squeeze(0).permute(1,2,0).detach().cpu().numpy()

    
    win_image_name = 'image'
    mask_image_name = 'mask'
    out_image_name = 'out'
    
    cv2.namedWindow(win_image_name)
    cv2.namedWindow(mask_image_name)
    cv2.namedWindow(out_image_name)
    
    cv2.moveWindow(win_image_name,800,1200)
    cv2.moveWindow(mask_image_name,1300,1200)
    cv2.moveWindow(out_image_name , 1800 , 1200)
    
    cv2.imshow(win_image_name,target)
    cv2.imshow(mask_image_name,mask)
    cv2.imshow(out_image_name,out_image)
    
    cv2.waitKey()
    cv2.destroyAllWindows()