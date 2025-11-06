# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- small UNet for segmentation (used earlier if needed) --
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec = DoubleConv(base*2, base)
        self.final = nn.Conv2d(base, out_ch, 1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = self.dec(torch.cat([d,e1],dim=1))
        return torch.sigmoid(self.final(d))

# -- SGCU activation --
class SGCU(nn.Module):
    """
    Sigmoidal Growing Cosine Unit (approx implementation).
    f(x) = sigmoid(a*x) * cos(b*x)  (a,b are learnable scales)
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        return torch.sigmoid(self.a * x) * torch.cos(self.b * x)

# -- Intermap Pooling block (grouped max across maps) --
class IntermapPool(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size
    def forward(self, x):
        # x: B x C x H x W
        B,C,H,W = x.shape
        g = self.group_size
        if C % g != 0:
            # pad channels
            pad = g - (C % g)
            x = F.pad(x, (0,0,0,0,0,pad))
            C = C + pad
        x = x.view(B, C//g, g, H, W)
        x, _ = torch.max(x, dim=2)  # max over group
        return x

# -- SGCIP-CNN classifier (compact) --
class SGCIP_CNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.sg = SGCU()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # Intermap pooling
            IntermapPool(group_size=2),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128* ( (32//4) * (32//4) ), 256), # assumes resized input 32x32 before classifier
            self.sg,
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        # make sure input is 1x32x32 or adapt via resize outside
        f = self.features(x)
        out = self.classifier(f)
        return out
