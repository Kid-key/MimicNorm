import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
]

# you need to download the models to ~/.torch/models
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
}

class MyScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, dZ):
        return dZ

class MyScaleLayer(nn.Module):
    def __init__(self):
        super(MyScaleLayer, self).__init__()
        #self.scale=nn.Parameter(torch.ones(1)*1.18)
    def forward(self, x):
        out=MyScale.apply(x)#*self.scale  #MyScale.apply(x)
        return out

class VGG(nn.Module):

    def __init__(self, features,class_bn=False, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        classifier = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ]
        if class_bn:
            # linearinitmode='fan_in'
            classifier+=[nn.BatchNorm1d(num_classes,affine=False)]
        self.classifier = nn.Sequential(*classifier)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.8 / n))
                if m.bias is not None:
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11_cbn(pretrained=False):
    if pretrained:
        print('no pretrained model!')
        assert False
    model = VGG(make_layers(cfg['A']), True)
    return model


def vgg11_bn(pretrained=False):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True))
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11_bn'])))
    return model



def vgg16_cbn(pretrained=False):
    if pretrained:
        print('no pretrained model!')
        assert False
    model = VGG(make_layers(cfg['D']), True)
    return model


def vgg16_bn(pretrained=False):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True))
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16_bn'])))
    return model


def vgg19_cbn(pretrained=False):
    if pretrained:
        print('no pretrained model!')
        assert False
    model = VGG(make_layers(cfg['E']), True)
    return model


def vgg19_bn(pretrained=False):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True))
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11_bn'])))
    return model

def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13'])))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13_bn'])))
    return model
