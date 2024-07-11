import torchinfo
from u_net_test import UNet
model = UNet(in_channels=3 , out_channels=1)
torchinfo.summary(model,input_size=(1,3,512,512))