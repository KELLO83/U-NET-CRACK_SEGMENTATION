import torchvision.transforms as T
import torch
import numpy as np

target = np.random.randint(0,255,size=(3,255,255))
target_t = T.Compose([
    T.ToTensor()
])


target_trans = target_t(target)
print(torch.min(target_trans),torch.max(target_trans))