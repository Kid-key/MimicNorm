"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import Layer

blocknum=0

def conv2d(i, o, kernel_size, stride=1, padding=0, groups=1):
    biasbool=kernel_size**2*i/groups>50
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=biasbool, groups=groups)

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        global blocknum

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.stride == 1 and self.in_channels == self.out_channels:
            blocknum += 1
            self.scale = nn.Parameter(torch.ones(1)*1.2**2/blocknum)
        else:
            self.scale = nn.Parameter(torch.ones(1)*1.2**2)
        self.residual = nn.Sequential(
            conv2d(in_channels, in_channels * t, 1),
            #Layer(in_channels * t),
            nn.ReLU(inplace=True),

            conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            #Layer(in_channels * t),
            nn.ReLU(inplace=True),

            conv2d(in_channels * t, out_channels, 1),
            #Layer(out_channels)
        )

    
    def forward(self, x):

        residual = self.residual(x)*self.scale

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        
        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            conv2d(3, 32, 1, padding=1),
            #Layer(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            conv2d(320, 1280, 1),
           # nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.conv2 = conv2d(1280, class_num, 1)
        self.lastbn=nn.BatchNorm1d(class_num,affine=False)  
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ss = max(2, m.in_channels/m.groups)
                n = m.kernel_size[0] * m.kernel_size[1] *ss
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
            
    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.lastbn(x)

        return x
    
    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        
        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()
