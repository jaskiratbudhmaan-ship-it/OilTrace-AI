import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        b=base_channels
        self.e1=DoubleConv(in_channels,b); self.p1=nn.MaxPool2d(2)
        self.e2=DoubleConv(b,b*2); self.p2=nn.MaxPool2d(2)
        self.e3=DoubleConv(b*2,b*4); self.p3=nn.MaxPool2d(2)
        self.e4=DoubleConv(b*4,b*8); self.p4=nn.MaxPool2d(2)
        self.bn=DoubleConv(b*8,b*16)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=DoubleConv(b*16,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2); self.d3=DoubleConv(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2); self.d2=DoubleConv(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2); self.d1=DoubleConv(b*2,b)
        self.out=nn.Conv2d(b,out_channels,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p1(e1)); e3=self.e3(self.p2(e2)); e4=self.e4(self.p3(e3))
        b=self.bn(self.p4(e4))
        d4=self.d4(torch.cat([self.u4(b),e4],1))
        d3=self.d3(torch.cat([self.u3(d4),e3],1))
        d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)
