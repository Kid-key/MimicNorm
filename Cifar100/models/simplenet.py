"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from layers import GradajustLayer,MyNormLayer,MyBatchNormLayer,MyBatchNormLayer1d
import math

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, class_bn=False, num_class=100):
        super().__init__()
        self.features = features

        classifier=[nn.Linear(512, 4096),nn.ReLU(inplace=True),nn.Dropout(),nn.Linear(4096, num_class)]
        # linearinitmode='fan_out'
        if class_bn:
            # linearinitmode='fan_in'
            classifier+=[nn.BatchNorm1d(num_class,affine=False)]
        else:
            classifier += [MyBatchNormLayer1d(num_class,momentum=0.1)]  
            # classifier+=[MyNormLayer()]
            # classifier+=[Linear()]
        self.classifier = nn.Sequential(*classifier)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                if m.in_channels==3:
                    m.weight.data.normal_(0, math.sqrt(20. / n))
                else:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()  

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output
class BN(nn.Module):

    def __init__(self, l, affine=False):
        super().__init__()

        self.bn = nn.BatchNorm2d(l, affine=affine)

    def forward(self, x):
        output = self.bn(x)    
        return output
        
def make_bn(l):
    bnlayer=nn.BatchNorm2d(l, affine=False)
    bn=nn.Sequential(bnlayer)
    setattr(bn,'bn',bnlayer)
    return bn

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        # if l == 'B':
        #     layers += [nn.BatchNorm2d(512)]
        #     continue
        # layers += [testlayer()]
        

        

        if batch_norm:
            bn=BN(l,True)
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1,bias=False)]
            layers += [bn]
            #layers += [MyBatchNormLayer(l,momentum=0.1)]             
            # layers +=[MyBatchNormLayer(l)]
            # layers +=[MyNormLayer()]
        else:
            # layers += [nn.Dropout(0.2)]
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1,bias=True)]
            #layers += [nn.Identity()]
            layers += [MyBatchNormLayer(l,momentum=0.1)]             
       
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def simnet_nobn():
    conf=[64, 'M', 128,'M', 256, 256,'M', 512, 512,'M', 512, 512, 'M'];
    return VGG(make_layers(conf))

def simnet():
    conf=[64, 'M', 128,'M', 256, 256,'M', 512, 512,'M', 512, 512, 'M'];
    return VGG(make_layers(conf, batch_norm=True))
    

def simnet_cbn():
    conf=[64, 'M', 128,'M', 256, 256,'M', 512, 512,'M', 512, 512, 'M'];
    return VGG(make_layers(conf),class_bn=True)


