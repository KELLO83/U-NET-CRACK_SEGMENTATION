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

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8,foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5,verbose=True,min_lr=1e-8
                                                           ,factor=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = DiceLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    # Resume
    if pretrained:
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt["best_score"]
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"Optimizer loaded from {opt.weights}")
            if start_epoch < opt.epochs:
                logging.info(
                    f"{opt.weights} has been trained for {start_epoch} epochs. Fine-tuning for {opt.epochs} epochs"
                )
        del ckpt
    
    print(f"opt.data {opt.data}")
    
    
    # Dataset
    train_data = RoadCrack(root=f'data/train', image_size=opt.image_size, mask_suffix="")
    test_data = RoadCrack(root=f'data/test', image_size=opt.image_size, mask_suffix="",train=False)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=8, drop_last=True, pin_memory=True)
    
    # import numpy as np
    # for i in train_loader:
    #     batch = i
    #     image = i[0] # tensor 1 3 512 512
    #     mask = i[1] # tensor 1 1 512 512
    #     print(batch[0].shape)
    #     print(batch[1].shape)
        
    #     target = image.squeeze(0).permute(1,2,0).detach().cpu().numpy() # 3 512 512
    #     mask = mask.permute(1,2,0).detach().cpu().numpy() 
        
    #     target = target * 255
    #     target = target.astype(np.uint8)
        
    #     mask = mask * 255
    #     mask = mask.astype(np.uint8)
        
    #     win_image_name = 'image'
    #     mask_image_name = 'mask'
        
    #     cv2.namedWindow(win_image_name)
    #     cv2.namedWindow(mask_image_name)
    #     cv2.moveWindow(win_image_name,1200,1200)
    #     cv2.moveWindow(mask_image_name,500,500)
        
    #     cv2.imshow(win_image_name,target)
    #     cv2.imshow(mask_image_name,mask)
    #     cv2.waitKey(1000)
        #cv2.destroyAllWindows()

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
                loss = criterion(output, target)

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(("%12s" * 2 + "%12.4g") % (f"{epoch + 1}/{opt.epochs}", mem, loss))

        dice_score, dice_loss = validate(model, test_loader, device)
        logging.info(f"VALIDATION: Dice Score: {dice_score:.4f}, Dice Loss: {dice_loss:.4f}")
        scheduler.step(dice_loss)
        
        ckpt = {
            "epoch": epoch,
            "best_score": best_score,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        
        torch.save(ckpt, last)
        if best_score < dice_score:
            best_score = max(best_score, dice_score)
            torch.save(ckpt, best)

    # Strip optimizers & save weights
    for f in best, last:
        strip_optimizers(f)



@torch.inference_mode()
def validate(model, data_loader, device ,conf_threshold=0.5):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()
    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
            if model.out_channels == 1:
                output = F.sigmoid(output) > conf_threshold
            dice_loss = criterion(output, target)
            dice_score += 1 - dice_loss
    model.train()

    return dice_score / len(data_loader), dice_loss


def parse_opt():
    parser = argparse.ArgumentParser(description="Crack Segmentation training arguments")
    parser.add_argument("--data", type=str, default="./data/compare", help="Path to root folder of data")
    parser.add_argument("--image_size", type=int, default=640, help="Input image size, default: 6") # 640 수정
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs, default: 50")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size, default: 12")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate, default: 1e-5")
    parser.add_argument("--weights", type=str, default="", help="Pretrained model, default: None")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")

    return parser.parse_args()


def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=opt.num_classes)

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
