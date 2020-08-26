"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from layers import MyNormLayer,MyBatchNormLayer,GradajustLayer
import math

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

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels,affine=True),
            # MyBatchNormLayer(out_channels),
            # MyNormLayer(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True)
            # MyBatchNormLayer(out_channels * BasicBlock.expansion),
            # MyNormLayer()
        )

        #shortcut
        # self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        # if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True)
                # MyBatchNormLayer(out_channels * BasicBlock.expansion),
                # MyNormLayer()
            )
        # self.record1=MyBatchNormLayer(out_channels * BasicBlock.expansion,momentum=1)
        # self.record2=MyBatchNormLayer(out_channels * BasicBlock.expansion,momentum=1)    
        
    def forward(self, x):
        x1=self.residual_function(x)
        # x1=self.record1(0.5*x1)
        x2=self.shortcut(x)
        # x2=self.record2(x2)
        return nn.ReLU(inplace=True)(0.5*x1+x2)

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 16
        self.basic_channels =16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False)
        # self.record00=MyBatchNormLayer(self.in_channels,momentum=1)
        self.bn0=nn.BatchNorm2d(self.in_channels,affine=True)
        # self.record01=MyBatchNormLayer(self.in_channels,momentum=1)
        self.relu0=nn.ReLU(inplace=True)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, self.basic_channels, num_block[0], 1)
        self.conv3_x = self._make_layer(block, self.basic_channels*2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, self.basic_channels*4, num_block[2], 2)
        #self.conv5_x = self._make_layer(block, self.basic_channels*4, num_block[3], 2)
        # self.aj1=GradajustLayer(1.5)
        # self.aj2=GradajustLayer(1.5)
        # self.aj3=GradajustLayer(1.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.basic_channels*4, num_classes)
        self.lastbn=nn.BatchNorm1d(num_classes)

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                if m.in_channels==3:
                    m.weight.data.normal_(0, math.sqrt(20. / n))
                else:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_uniform_(m.conv1.weight,a=2.0)
                # nn.init.kaiming_uniform_(m.conv2.weight,a=2.0)
                # nn.init.constant_(m.conv2.weight, 0)
            # if isinstance(m, BottleNeck):
            #     # nn.init.kaiming_normal_(m.conv1.weight,nonlinearity='relu')
            #     nn.init.kaiming_uniform_(m.conv1.weight,a=3.0)
            #     nn.init.kaiming_uniform_(m.conv2.weight,a=3.0)
            #     nn.init.kaiming_uniform_(m.conv3.weight,a=3.0)

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
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        # output = self.record00(output)
        output = self.bn0(output)
        # output = self.record01(output)
        output = self.relu0(output)
        output = self.conv2_x(output)
        # output = self.aj2(output)
        output = self.conv3_x(output)
        # output = self.aj3(output)
        output = self.conv4_x(output)
        #output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.lastbn(output)

        return output 

def simresnet():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2])







