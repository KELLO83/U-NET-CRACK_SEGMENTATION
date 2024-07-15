import torchinfo
from crackseg.models.unet import UNet
from crackseg.models.u_net_paper import UNet_Paper

model = UNet(in_channels=3,out_channels=2)
#model__ = UNet_Paper()
torchinfo.summary(model,input_size=(1,3,512,512))
#torchinfo.summary(model__,input_size=(1,3,512,512))