import os
import logging
import argparse
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn 
from torch.utils.data import DataLoader
from crackseg.models.resuidual_unet import ResUNet
from torch.utils.tensorboard import SummaryWriter

from crackseg.models import UNet
from crackseg.utils.dataset import CustomDataset,RoadCrack
from crackseg.utils.general import random_seed
from crackseg.utils.losses import CrossEntropyLoss, DiceCELoss, DiceLoss, FocalLoss
from custom_loss import BCEDiceLoss , FocalLoss ,CustomCrossEntropy
import torchmetrics
import os
import numpy as np
import cv2
import pdb
import torchinfo

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'



def visualize_batch_cv2(images, masks):
    batch_size = images.size(0)
    for i in range(batch_size): # [4,3,512,512] , [4,1,512,512]
        image = images[i].permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        image = (image * 255).astype(np.uint8)  # 이미지 값을 0-255 범위로 변환
        mask = masks[i].squeeze(0).cpu().numpy()  # [1, H, W] -> [H, W]
        mask = (mask * 255).astype(np.uint8)  # 마스크 값을 0-255 범위로 변환
        
        # OpenCV를 사용하여 이미지와 마스크 표시
        image_named = "image"
        mask_named = "mask"
        cv2.namedWindow(image_named)
        cv2.namedWindow(mask_named)
        cv2.moveWindow(image_named,1000,1000)
        cv2.moveWindow(mask_named,1500,1000)        
        cv2.imshow(image_named, image)
        cv2.imshow(mask_named, mask)
        cv2.waitKey(1000)  # 키 입력을 대기 (무한 대기)
        cv2.destroyAllWindows()
    
        
def strip_optimizers(f: str) -> None:
    """Strip optimizer from 'f' to finalize training"""
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "best_score":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # get file size
    logging.info(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")


def train(opt, model, device):
    best_score, start_epoch = 0, 0
    best, last = f"{opt.save_dir}/best.pt", f"{opt.save_dir}/last.pt"

    # Check pretrained weights
   
    pretrained = opt.weights.endswith(".pt")

    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
        logging.info(f"Model ckpt loaded from {opt.weights}")
    model.to(device)

    # Optimizers & LR Scheduler & Mixed Precision & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-3,foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10,verbose=True,min_lr=1e-12
                                                           ,factor=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = BCEDiceLoss()
    writer = SummaryWriter('./log_dir')
    
    # Resume
    if pretrained:
        print("======================pretrained=======================")
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"Optimizer loaded from {opt.weights}")
            if start_epoch < opt.epochs:
                logging.info(
                    f"{opt.weights} has been trained for {start_epoch} epochs. Fine-tuning for {opt.epochs} epochs"
                )
        del ckpt
    
    
    # #dataset
    # image_path = 'data/CRKWH100_IMAGE'
    # mask_path = 'data/CRKWH100_MASK'
    # image_type = 'png'
    
    # test_image__path = 'data/CRKWH100_IMAGE/val'
    # test_mask_path = 'data/CRKWH100_MASK/val'
    # ' data/CrackTree_IMAGE ' 
    # image_path = 'data/CRKWH100_IMAGE'

    image_path = opt.data
    t_ = image_path.split('/')[1].split('_')
    t__ =  f"{image_path.split('/')[0]}/{t_[0]+'_MASK'}"
    
    
    mask_path = t__
    image_type = 'jpg'
    
    test_image_path = os.path.join(opt.data + '/val')
    test_mask_path = os.path.join(t__ + '/val')
    
    train_data = CustomDataset(image_path , mask_path , image_type , is_train=True) # crackTree dataset
    test_data = CustomDataset(test_image_path,test_mask_path,image_type,is_train=True)
    
    it_test = iter(train_data)
    it_next = it_test.__next__()
    it_mask = it_next[1]
    
    #DataLoader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True) # batch_size = opt.batch_size
    test_loader = DataLoader(test_data,batch_size=1,num_workers=8,shuffle=False, pin_memory=True)
        
    count = 0    
    for i in train_loader:
        batch = i
        image = batch[0]
        mask = batch[1]
        visualize_batch_cv2(image , mask)
        count += 1
        break
 
    best_loss = 100
    scalar_count = 0
    val_count = 0

    # Training
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0
        logging.info(("\n" + "%12s" * 3) % ("Epoch", "GPU Mem", "Loss"))
        
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for image, target in progress_bar:
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(image)
                loss = criterion(output,target)
                writer.add_scalar('loss/train',loss.item(),scalar_count)
                scalar_count += 1
                
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer) 
            grad_scaler.update()

            epoch_loss += loss.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(("%12s" * 2 + "%12.4g") % (f"{epoch + 1}/{opt.epochs}", mem, loss))

        loss = validate(model, test_loader, device)
        logging.info(f"VALIDATION: Loss: {loss:.4f}")
        scheduler.step(loss)
        print(f"{scheduler.get_last_lr()[0]:.2e}")
        writer.add_scalar("loss/val",loss.item(),val_count)
        val_count += 1
        ckpt = {
            "epoch": epoch,
            "best_loss": loss,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        
        torch.save(ckpt, last)
        if best_loss > loss:
            best_loss = min(best_loss, loss.item())
            torch.save(ckpt, best)
        

        
    # Strip optimizers & save weights
    for f in best, last:
        strip_optimizers(f)
    
    writer.close()



@torch.inference_mode()
def validate(model, data_loader, device):
    model.eval()
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = FocalLoss()
    criterion = BCEDiceLoss()
    #criterion =CustomCrossEntropy()
    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)
    model.train()
    return loss


def parse_opt():
    parser = argparse.ArgumentParser(description="Crack Segmentation training arguments")
    parser.add_argument("--data", type=str, default="data/CrackTree_IMAGE", help="Path to root folder of data")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size, default: 6") # 512 수정
    parser.add_argument("--save-dir", type=str, default="weights/stone/bcedice", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs, default: 50")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size, default: 12")
    parser.add_argument("--lr", type=float, default=1e-8, help="Learning rate, default: 1e-5")
    parser.add_argument("--weights", type=str, default='', help="Pretrained model, default: None")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")

    return parser.parse_args()


def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = ResUNet(in_channels=3, out_channels=opt.num_classes)
    torchinfo.summary(model,input_size=(1,3,opt.image_size,opt.image_size))
    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    random_seed()
    # Create folder to save weights
    os.makedirs(opt.save_dir, exist_ok=True)

    train(opt, model, device)


if __name__ == "__main__":
    params = parse_opt()
    main(params)
