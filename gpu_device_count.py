import torch

print(torch.cuda.get_device_name(device = 0))
print(torch.cuda.device_count())