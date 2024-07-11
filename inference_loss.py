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

@torch.inference_mode
def evaulte(model , data_loader,device):
    metric = BinaryF1Score().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    average_precision = torchmetrics.AveragePrecision(task='binary')
    ap_result = []
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    for image , target  in tqdm(data_loader , total=len(data_loader)):
        image , target = image.to(device) , target.to(device)
        target = target / 255.0 # CRACK 500 DATASET 사용시 삭제 필요 
        with torch.no_grad():
            output = model(image)
            if model.out_channels == 1:
                output = F.sigmoid(output) > 0.5
            dice_loss = criterion(output,target)
            dice_score += 1 - dice_loss
            metric.update(output[:,0,:,:],target)
            precision.update(output[:,0,:,:],target)
            recall.update(output[:,0,:,:],target)
            iou_metric.update(output[:,0,:,:],target)   
            
            target_float64 = target.to(torch.int64)
            ap_ = average_precision(output[:,0,:,:],target_float64)     
            ap_result.append(ap_)
            
    f1_score = metric.compute()
    precision_result = precision.compute()
    recall_result = recall.compute()
    iou_score_result = iou_metric.compute()
    ap_average = sum(ap_result) / len(ap_result)
    model.train()
    
    return dice_score / len(data_loader) , dice_loss , f1_score , precision_result , recall_result , iou_score_result , ap_average


@torch.inference_mode
def image_segmentation_runnung(model , input , device):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    with torch.no_grad():
        image,mask = input[0].to(device) , input[1].to(device)
        mask = mask / 255.0 # CRACK 500 아닐시 사용
        output = model(image)
        dice_loss = criterion(output , mask)
        
    return output , dice_loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=1)
    weith_path = 'weights/CrackLS315/best.pt'
    ckpt = torch.load(weith_path,map_location=device)
    model.load_state_dict(ckpt['model'].float().state_dict())
    model = model.to(device)
    
    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    
    image_path = 'data/CrackLS315_IMAGE/test'
    mask_path = 'data/CrackLS315_MASK/test'
    test_dataset = CustomDataset(image_path,mask_path,'jpg',is_resize=True)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=8,shuffle=False,pin_memory=True)
    
    for index , i in enumerate(test_loader):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        if index==2:
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
    
    ds , dl , score , pre , rec , iou , ap_res = evaulte(model,test_loader,device)
    print("============== Total Test set\nDice socre : {:.5f}\n Dice loss : {:.5f}\n F1 Score : {:.5f}\n Precision : {:.5f}\n Recall : {:.5f}\n IoU : {:.5f}\nAP : {:.5f}\n====================".
          format(ds,dl,score*100,pre*100,rec*100,iou*100,ap_res))
    
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
    label_np = label_np * 255
    target = np.clip(result_np[1] * (-1) * 255,0,255)
    #cv2.imshow("Channel 1",result_np[0])
    cv2.imshow("CHannel 2",target)

    cv2.imshow("orgin image",image_np)
    cv2.imshow("label_np",label_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
    