"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import numpy as np
import math
from layers import Layer,MyScaleLayer

class MyPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, dZ):
        return dZ#sqrt(1.4)


class MyPassLayer(nn.Module):
    def __init__(self,inival=1.2):
        super(MyPassLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(1)*inival)
    def forward(self, x):
        out=x*self.scale
        return out

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        global subnetnum
        subnetnum += 1.
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            #MyPassLayer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
            #MyPassLayer(out_channels * BasicBlock.expansion),
           # MyScaleLayer(0.1),
        )
        self.scale =nn.Parameter(torch.ones(1)/subnetnum*1.2**2)
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True),
                MyPassLayer() # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x)*self.scale + self.shortcut(x))

subnetnum=0    
class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            #MyPassLayer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=True),
            #MyPassLayer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=True),
            #MyPassLayer(out_channels * BottleNeck.expansion),
            #MyScaleLayer(0.1)
        )
        global subnetnum
        subnetnum += 1.
        self.scale = nn.Parameter(torch.ones(1)/subnetnum*1.2**3)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                MyPassLayer()
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x)*self.scale + self.shortcut(x))
        
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, CBN=True):
        super().__init__()
        global subnetnum
        subnetnum =0
        self.in_channels = 64
        self.CBN=CBN
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
#            MyPassLayer(64),
            nn.ReLU(inplace=True))
        self.bias1 = nn.Parameter(torch.zeros(1))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
     #   self.bn=nn.BatchNorm2d(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if CBN:
            self.lastbn=nn.BatchNorm1d(num_classes,affine=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if not m.bias is None:
                    #m.bias.data.normal_(0, math.sqrt(0.0 / m.in_channels))
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
      #  for m in self.modules():
       #     if isinstance(m, BottleNeck):
         #       nn.init.constant_(m.conv3.weight, 0)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
            #self.inplanes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
       # output=self.bn(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.CBN:
            output = self.lastbn(output)   
        #if self.CBN:
       #     output = output1

        return output 

def resnet18_cbn():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet18_nobn():
    """ return a ResNet 18 object
    """
    print('resnet18: no CBn.')
    return ResNet(BasicBlock, [2, 2, 2, 2],CBN=False)

def resnet34_cbn():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50_cbn():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet50_nobn():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],CBN=False)

def resnet101_cbn():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



