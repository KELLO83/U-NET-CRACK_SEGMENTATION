import torchinfo
from crackseg.models import UNet

model = UNet(in_channels=3 , out_channels=1)
torchinfo.summary(model,input_size=(16,3,360,640))