import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2

class PadToSquare:
    def __call__(self, image, mask):
        """ 640 * 640 image pad"""

        w, h = image.size
        
        # 패딩 크기 계산 (상하 좌우에 고르게 패딩을 추가하여 정사각형으로 만듦)
        pad_left = (640 - w) // 2
        pad_right = 640 - w - pad_left
        pad_top = (640 - h) // 2
        pad_bottom = 640 - h - pad_top
        
        # 패딩 적용
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        image = F.pad(image, padding, fill=0)
        mask = F.pad(mask, padding, fill=0)
        
        return image, mask
