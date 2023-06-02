import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as Tfs
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.cnn_reshaper = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        self.network = nn.Sequential(
            nn.Conv2d(in_size, mid_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(),
            
            nn.Conv2d(mid_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )
    
    def forward(self, xb):
        if self.in_size != self.out_size:
            xb_reshaped = self.cnn_reshaper(xb)
        else:
            xb_reshaped = xb
        return self.network(xb) + xb_reshaped

class CNNBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            #extra
            ResBlock(64, 64, 64),
            ResBlock(64, 128, 64),
            #extra
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            ResBlock(128, 128, 128),
            ResBlock(128, 256, 128),
        
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            ResBlock(512, 512, 512),
            ResBlock(512, 1024, 512),
        
            nn.MaxPool2d(5,4),
            nn.Flatten()
        )
        
    def forward(self, xb):
        return self.network(xb)

class Encoder(nn.Module):
    def __init__(self, z_d=256):
        super().__init__()
        self.backbone = CNNBackBone()  
        self.logvar_branch = nn.Linear(512,z_d)
        self.mean_branch = nn.Linear(512,z_d)
        self.z_d = z_d
    def forward(self, xb):
        backbone_out = self.backbone(xb)
        logvar = self.logvar_branch(backbone_out)
        mean = self.mean_branch(backbone_out)
        
        ##simulate sampling from distributions determined by dimensions of mean and logvar
        ##where std = torch.exp(0.5*logvar) - reparameterization trick
        std = torch.exp(0.5*logvar)
        z = mean + (std * torch.randn(logvar.shape[0],logvar.shape[1]).to(device))
        return z, logvar, mean, std

class Decoder(nn.Module):
    def __init__(self, z_d=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_d, 512),
        )
        self.network = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResBlock(128,128,128),
                ResBlock(128,256,128),
                #
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResBlock(128,128,128),
                ResBlock(128,128,64),
                #
            
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResBlock(64,64,64),
                ResBlock(64,64,32),
                #
            
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ResBlock(32,32,32),
                ResBlock(32,32,32),
            
                nn.Upsample(scale_factor=1.40625, mode='bilinear', align_corners=True),
                ResBlock(32,16,16),
                ResBlock(16,16,3),
                nn.Sigmoid()
            
#                 nn.ConvTranspose2d(128, 128, stride=2, kernel_size=4, padding=1, bias=False),
#                 nn.BatchNorm2d(128),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 #
#                 nn.ConvTranspose2d(128, 64, stride=2, kernel_size=4, padding=1, bias=False),
#                 nn.BatchNorm2d(64),
#                 nn.LeakyReLU(0.1, inplace=True),
            
#                 nn.ConvTranspose2d(64, 32, stride=2, kernel_size=4, padding=1, bias=False),
#                 nn.BatchNorm2d(32),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 #
#                 nn.ConvTranspose2d(32, 3, stride=2, kernel_size=4, padding=1),
#                 nn.Sigmoid()
        )
    def forward(self, z):
        linear_out = self.linear(z)
        return self.network(linear_out.view(-1,128,2,2))

class VAE(nn.Module):
    def __init__(self, z_d=256):
        super().__init__()
        self.encoder = Encoder(z_d=z_d)
        self.decoder = Decoder(z_d=z_d)
        self.z_d = z_d
    def forward(self, xb):
        z, logvar, mean, std = self.encoder(xb)
        pred = self.decoder(z)
        return pred, z, logvar, mean, std