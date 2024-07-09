from typing import Any
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import numpy as np

class pdImage:
    def __call__(self, image, mask=False):
        """ 640 * 640 image pad """
        # 이미지 크기 가져오기
        if mask:
            h, w = image.shape
            image = image.astype(np.uint8) * 255
            # 패딩 크기 계산
            pad_left = (640 - w) // 2
            pad_right = 640 - w - pad_left
            pad_top = (640 - h) // 2
            pad_bottom = 640 - h - pad_top

            # 패딩 적용 (2D)
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))
            image = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            h, w, c = image.shape
            # 패딩 크기 계산
            pad_left = (640 - w) // 2
            pad_right = 640 - w - pad_left
            pad_top = (640 - h) // 2
            pad_bottom = 640 - h - pad_top

            # 패딩 적용 (3D)
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            image = np.pad(image, padding, mode='constant', constant_values=0)
        
        return image

class ResizeImage():
    def __init__(self) -> None:
        self.base_transform = T.Compose([
            T.CenterCrop(448)
        ])
    def __call__(self,image , mask) -> Any:
        image = Image.open(image)
        mask = Image.open(mask)
        
        if image is None or mask is None:
            raise FileExistsError
        

        image_trans = self.base_transform(image)
        mask_trans = self.base_transform(mask)
        
        
        return image_trans , mask_trans


if __name__ == "__main__":
    image_path = 'assets/20160222_114759_641_721.jpg'
    mask_path = 'assets/20160222_114759_641_721.png'
    image = Image.open(image_path)
    mask  = Image.open(mask_path)
        
    image_array = np.array(image)
    mask_array = np.array(mask)
    
    pdt = pdImage()
    pd_image = pdt(image_array)
    pd_mask = pdt(mask_array,mask=True)
    
    cv2.imshow("orgin_image",image_array)
    cv2.imshow("pd_image",pd_image)
    cv2.imshow("mask_array",mask_array.astype(np.uint8) * 255)
    cv2.imshow("pd_mask",pd_mask)
    
    cv2.imwrite('pad_image.jpg',pd_image)
    cv2.imwrite('pd_mask.png',pd_mask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("==================The END =======================")
    
    
        