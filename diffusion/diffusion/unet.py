import torch.nn as nn
import torch
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

################################################# Vanilla UNet ####################################################

class ConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownSampleBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=0):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.downsample(x)
    

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)

    def center_crop(self, enc_feature, dec_feature):
        _, _, H, W = enc_feature.size()
        _, _, h, w = dec_feature.size()
        w_start = (W - w) // 2
        h_start = (H - h) // 2
        w_end = w_start + w
        h_end = h_start + h
        return enc_feature[:, :, h_start:h_end, w_start:w_end]
    
    def forward(self, x, enc_feature):
        upsampled = self.upsample(x)
        crop_copy = self.center_crop(enc_feature, upsampled)
        concat = torch.cat([upsampled, crop_copy], dim=1)
        return self.conv(concat)
    

class UNet(nn.Module):
    """https://arxiv.org/pdf/1505.04597"""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 64)
        self.down1 = DownSampleBlock(64, 128)
        self.down2 = DownSampleBlock(128, 256)
        self.down3 = DownSampleBlock(256, 512)
        self.down4 = DownSampleBlock(512, 1024)
        self.up1 = UpSampleBlock(1024, 512)
        self.up2 = UpSampleBlock(512, 256)
        self.up3 = UpSampleBlock(256, 128)
        self.up4 = UpSampleBlock(128, 64)
        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x) # (568, 568, 64)
        x2 = self.down1(x1) # (280, 280, 128)
        x3 = self.down2(x2) # (136, 136, 256)
        x4 = self.down3(x3) # (64, 64, 512)
        x5 = self.down4(x4) # (28, 28, 1024)
        x = self.up1(x5,x4) # (52, 52, 512)
        x = self.up2(x, x3) # (100, 100, 256)
        x = self.up3(x, x2) # (196, 196, 128)
        x = self.up4(x, x1) # (388, 388, 64)
        x = self.out(x) #  (388, 388, 2)

        return x


################################################ UNet with TimeEmbedding  ###############################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """given (B, 1), return (B, dim)"""
        half_dim = self.dim // 2
        emb = torch.log(torch.ones(1,device=DEVICE) *self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim,device=DEVICE) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """project (B, dim) to (B, time_dim)"""
    def __init__(self, time_dim:int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

    def forward(self, t):
        return self.time_mlp(t)
    

class ConvBlockWithTime(ConvBlock):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1, time_dim:int=256):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.time_embd = nn.Sequential(
            nn.Linear(time_dim, in_channels),
            nn.GELU()
        )
        
        
    def forward(self, x, t=None):
        if t is not None:
            t = self.time_embd(t)
            t = t[:, :, None, None]
            x = x + t
        return self.double_conv(x)
    

class DownSampleBlockWithTime(DownSampleBlock):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1,  time_dim:int=256):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.time_embd = nn.Sequential(
            nn.Linear(time_dim, in_channels),
            nn.GELU()
        )
    
    def forward(self, x, t=None):
        if t is not None:
            t = self.time_embd(t)
            t = t[:, :, None, None]
            x = x + t
        return super().forward(x)
    

class UpSampleBlockWithTime(UpSampleBlock):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1,  time_dim:int=64):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.time_embd = nn.Sequential(
            nn.Linear(time_dim, in_channels),
            nn.GELU()
        )
    
    def forward(self, x, enc_feature, t=None):
        if t is not None:
            t = self.time_embd(t)
            t = t[:, :, None, None]
            x = x + t
        return super().forward(x, enc_feature)
    
class UNetWithTime(UNet):

    def __init__(self, init_channels:int=128, time_dim:int=256):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=init_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_channels),
            nn.GELU()
        )
        self.time_embd = TimeEmbedding(time_dim)
        self.mlp = nn.Linear(time_dim,init_channels)
        self.conv1 = ConvBlockWithTime(init_channels, 64, time_dim=time_dim)
        self.down1 = DownSampleBlockWithTime(64, 128, time_dim=time_dim)
        self.down2 = DownSampleBlockWithTime(128, 256, time_dim=time_dim)
        self.down3 = DownSampleBlockWithTime(256, 512, time_dim=time_dim)
        self.down4 = DownSampleBlockWithTime(512, 1024, time_dim=time_dim)
        self.up1 = UpSampleBlockWithTime(1024, 512, time_dim=time_dim)
        self.up2 = UpSampleBlockWithTime(512, 256, time_dim=time_dim)
        self.up3 = UpSampleBlockWithTime(256, 128, time_dim=time_dim)
        self.up4 = UpSampleBlockWithTime(128, 64, time_dim=time_dim)
        self.out = nn.Sequential(
            nn.Conv2d(64,1,1),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):          
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        if t is not None:
            t = self.time_embd(t)
            t_embd = self.mlp(t)[:, :, None, None] #(1, 1, 256)
            x = x + t_embd
        x1 = self.conv1(x, t) # (568, 568, 64)

        x2 = self.down1(x1, t) # (280, 280, 128)
        x3 = self.down2(x2, t) # (136, 136, 256)
        x4 = self.down3(x3, t) # (64, 64, 512)
        x5 = self.down4(x4, t) # (28, 28, 1024)
        x = self.up1(x5, x4, t) # (52, 52, 512)
        x = self.up2(x, x3, t) # (100, 100, 256)
        x = self.up3(x, x2, t) # (196, 196, 128)
        x = self.up4(x, x1, t) # (388, 388, 64)
        x = self.out(x) #  (388, 388, 1)


        
        return x
    

