import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data

from crackseg.utils.general import TrainTransforms
from image_pad import PadToSquare
import pdb

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

        # TODO: letterbox or some other resizing methods should be used if image is not square.
        # resize input
        # image = image.resize((self.image_size, self.image_size))
        # mask = mask.resize((self.image_size, self.image_size))

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
