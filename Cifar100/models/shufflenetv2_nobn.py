"""shufflenetv2 in pytorch



[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)
    
def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class MyScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,scale):
        ctx.scale=scale
        return input*scale
    @staticmethod
    def backward(ctx, dZ):
        return dZ*ctx.scale,dZ.mean()

class MyScaleLayer(nn.Module):
    def __init__(self, initvalue=1.2):
        super(MyScaleLayer, self).__init__()
        self.scale=nn.Parameter(torch.ones(1)*initvalue)
    def forward(self, x):
        out = x.mul(self.scale)  #MyScale.apply(x)
        return out

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(

                nn.Conv2d(in_channels, in_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                MyScaleLayer(1.2**2),
                nn.ReLU(inplace=True)              
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                MyScaleLayer(),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                
                nn.Conv2d(in_channels, in_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                MyScaleLayer(1.2**2),
                nn.ReLU(inplace=True),
            )
           
    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x
        
        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)*self.scale+ self.bias
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)
        
        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, class_num=100,cbn=True):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')
        
        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1)
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            MyScaleLayer(),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], class_num)
        self.cbn=cbn
        if cbn:
            self.lastbn=nn.BatchNorm1d(class_num,affine=False)  
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ss = max(4, m.in_channels/m.groups)
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
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.cbn:
            x = self.lastbn(x)
        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1
        
        return nn.Sequential(*layers)

def shufflenetv2_cbn():
    return ShuffleNetV2()

def shufflenetv2_nobn():
    return ShuffleNetV2(cbn=False)




