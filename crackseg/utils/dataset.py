import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import torch.utils
import torch.utils.data

from crackseg.utils.general import TrainTransforms
from image_pad import PadToSquare
import pdb
import natsort

import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import torch

from crackseg.utils.general import TrainTransforms
from image_pad import PadToSquare
import pdb
import natsort


from torchvision import transforms as T
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms

class CustomHorizontalFlip:
    def __call__(self, image, mask):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        if random.random() > 0.5:
            image = TF.hflip(image)
        random.seed(seed)
        if random.random() > 0.5:
            mask = TF.hflip(mask)
        return image, mask


class CustomGaussianNoise:
    def __init__(self, probability=0.3, mean=0.0, std=5.0):
        self.probability = probability
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = self.add_gaussian_noise(image)
        return image, mask

    def add_gaussian_noise(self, image):
        np_image = np.array(image)
        noise = np.random.normal(self.mean, self.std, np_image.shape)
        np_noisy_image = np_image + noise
        np_noisy_image = np.clip(np_noisy_image, 0, 255).astype(np.uint8)
        return TF.to_pil_image(np_noisy_image)

    
class CustomDataset(torch.utils.data.Dataset):
    """CRKWH1000 CRACKLS315 STONE331 CrackTree260"""
    
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str, 
                 image_type: str, 
                 image_size: int = 512, 
                 is_stone: bool = False,
                 is_train : bool = False) -> None:
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_type = image_type
        self.image_size = image_size
        self.is_stone = is_stone
        self.is_train = is_train #  Crack Tree 260 
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])
        
        self.image_filenames = self._load_filenames(image_dir, image_type)
        self.mask_filenames = self._load_filenames(mask_dir, ".bmp")
        
        self.aug = CustomHorizontalFlip()
        self.aug2 = CustomGaussianNoise()
        
        if not self.image_filenames:
            raise FileNotFoundError(f"Files not found in {image_dir}")
        
        if not self.mask_filenames:
            raise FileNotFoundError(f"Files not found in {mask_dir}")

    def _load_filenames(self, dir_path, file_extension):
        filenames = [os.path.splitext(filename)[0] for filename in os.listdir(dir_path) if filename.endswith(file_extension)]
        return natsort.natsorted(filenames)
    
    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, f"{filename}.{self.image_type}")
        mask_path = os.path.join(self.mask_dir, f"{filename}.bmp")
        
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if image is None or mask is None:
            raise FileNotFoundError("Exception: File not Found")
        
        if self.is_train:
            image = self.__resize_and_pad(image)
            mask = self.__resize_and_pad(mask)
            
        image, mask = self.aug(image, mask)
        image, mask = self.aug2(image, mask)
        
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
                        
        if self.is_stone:
            image = image.unsqueeze(dim=0)
            image = F.interpolate(image, size=(512,512), mode='nearest')
            image = image.squeeze(dim=0)

        
        assert image.shape[1:3] == mask.shape[1:3], f"`image`: {image.shape[1:3]} and `mask`: {mask.shape[1:3]} are not the same"
        
        return image, mask

    def __resize_and_pad(self , image:Image , size=(512, 512)):
        image = np.array(image)
        
        h, w = image.shape[:2]
        scale = size[0] / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        delta_w = size[1] - new_w
        delta_h = size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0] 
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        print("resize image :",padded_image.shape)
        
        image = Image.fromarray(padded_image)
        return padded_image



def to_binary(mask_image):
    # Convert PIL Image to numpy array
    mask_array = np.array(mask_image)

    # Apply threshold directly to create a binary image with values 0 and 255
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8) / 255

    # Convert numpy array back to PIL Image, already in binary format
    binary_mask = Image.fromarray(binary_mask)

    return binary_mask

class RoadCrack(data.Dataset):
    def __init__(
            self,
            root: str,
            image_size: int = 448,
            transforms: TrainTransforms = TrainTransforms,
            mask_suffix: str = "_mask",
            train:bool = True
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root)) if filename.endswith(".jpg")]
        
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")
        self.transforms = transforms()
        self.padding = PadToSquare()
        self.is_train = train 
        
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        if self.is_train:
            image_path = os.path.join(self.root, f"{filename}.jpg")
            mask_path = os.path.join(self.root, f"{filename}.png")
        else :
            image_path = os.path.join(self.root,f"{filename}.jpg")
            mask_path = os.path.join(self.root,f"{filename}_mask.png")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if image is None or mask is None:
            raise FileNotFoundError("Exception FIle not Founded")
        # mask = to_binary(mask)
        image , mask = self.padding(image,mask)

        
        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        # transform
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask


def to_binary(mask_image):
    # Convert PIL Image to numpy array
    mask_array = np.array(mask_image)

    # Apply threshold directly to create a binary image with values 0 and 255
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8) / 255

    # Convert numpy array back to PIL Image, already in binary format
    binary_mask = Image.fromarray(binary_mask)

    return binary_mask
