import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


image = Image.open("11.jpg")
image_ = np.array(image)

angle = 45
transformed_image = TF.affine(image, angle=angle, translate=(0, 0), scale=1.0, shear=(0, 0))

cv2.imshow("orgin image",image_)
cv2.imshow("transformed image",np.array(transformed_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
