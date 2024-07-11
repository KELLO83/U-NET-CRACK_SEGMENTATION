import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        self.encoder5 = nn.Sequential(
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024)
        )
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 512)
        )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )
        
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        e5 = self.encoder5(p4)
        
        # Decoding path
        d4 = self.upconv4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        
        return out

# Example usage
model = UNet()
torchinfo.summary(model,input_size=(1,3,512,512))


