import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import torch
from crackseg.models import UNet
import cv2

class pdImage:
    def __call__(self, image):
        """ 640 * 640 image pad"""
        # 이미지 크기 가져오기
        _ , h, w = image.size()
        
        # 패딩 크기 계산 (상하 좌우에 고르게 패딩을 추가하여 정사각형으로 만듦)
        pad_left = (640 - w) // 2
        pad_right = 640 - w - pad_left
        pad_top = (640 - h) // 2
        pad_bottom = 640 - h - pad_top
        
        # 패딩 적용
        padding = (pad_left, pad_top, pad_right, pad_bottom) #provided this is the padding for the left, top, right and bottom borders respectively
        image = F.pad(image, padding, fill=0)
        
        return image 

def preprocess(image, is_mask):
    """Preprocess image and mask"""
    img_ndarray = np.asarray(image)
    if not is_mask:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

    return img_ndarray

@torch.inference_mode
def predict(model, image, device, conf_thresh):
    model.eval()
    model.to(device)
    
    # Preprocess
    #pdt = pdImage()
    image = torch.from_numpy(preprocess(image, is_mask=False))
    #image = pdt(image)
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image).cpu()
        if model.out_channels > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > conf_thresh

    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray):
    """Convert mask to image"""
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def parse_opt():
    parser = argparse.ArgumentParser(description="Crack Segmentation inference arguments")
    parser.add_argument("--weights", default="./weights/best.pt", help="Path to weight file (default: best.pt)")
    parser.add_argument("--input", type=str, default="assets/pad_image.jpg", help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Path to save mask image")
    parser.add_argument("--view", action="store_true", help="Visualize image and mask",default='True')
    parser.add_argument("--no-save", action="store_true", help="Do not save the output masks")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold for mask") # mask threshold

    return parser.parse_args()


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(opt.weights):
        ckpt = torch.load(opt.weights, map_location=device)
    else:
        raise AssertionError(f"Trained weights not found in {opt.weights}")
    
    # Initialize model and load checkpoint
    model = UNet(in_channels=3, out_channels=2)
    model.load_state_dict(ckpt["model"].float().state_dict())

    # Load & Inference
    image = Image.open(opt.input)
    label = Image.open("assets/pd_mask.png")
    
    output = predict(model=model, image=image, device=device, conf_thresh=opt.conf_thresh)

    # Convert mask to image
    result = mask_to_image(output)
    result.save(opt.output)

    result_np_array = np.array(result)
    image_array = np.array(image)
    # plt.imshow(image_array)
    # plt.axis('off')
    # plt.show()
    label_array = np.array(label)
    
    # Visualize
    if opt.view:
        print("show start")
        cv2.imshow("image_array",image_array)
        cv2.imshow('label',label_array)
        cv2.imshow("result_np_array",result_np_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("=====================END========================")
    
if __name__ == "__main__":
    params = parse_opt()
    main(params)
