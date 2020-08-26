""" helper function

author baiyu
"""

import sys

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

###################https://zhuanlan.zhihu.com/p/49329030################
import torch.nn as nn
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            if c==None:
                m._modules[name] = DummyModule()
                continue
            try:
                bc = fuse(c, child)
                #print('+1')
            except Exception as e:
                print(e)
                print(c)
                print(child)
                assert False
            
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)

                
##################################################################
def get_network(netname, use_gpu=True):
    """ return given network
    """
               
    if netname== 'vgg16':
        from models.vgg import vgg16_bn #!
        net = vgg16_bn()
    elif netname== 'vgg16_cbn':
        from models.vgg_nobn import vgg16_cbn #!
        net = vgg16_cbn()
    elif netname== 'vgg11':
        from models.vgg import vgg11_bn #!
        net = vgg11_bn()
    elif netname== 'vgg11_cbn':
        from models.vgg_nobn import vgg11_cbn #!
        net = vgg11_cbn()
    elif netname== 'vgg11_nobn':
        from models.vgg_nobn import vgg11_nobn #!
        net = vgg11_nobn()
    elif netname== 'vgg16_nobn':
        from models.vgg_nobn import vgg16_nobn #!
        net = vgg16_nobn()
    elif netname== 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif netname== 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif netname== 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif netname== 'resnet18_nobn':
        from models.resnet_nobn import resnet18_nobn
        net = resnet18_nobn()
    elif netname== 'resnet18_fixup':
        from models.resnet_fixup import resnet18
        net = resnet18()
    elif netname== 'resnet50_fixup':
        from models.resnet_fixup import resnet50
        net = resnet50()
    elif netname== 'resnet18_cbn':
        from models.resnet_nobn import resnet18_cbn
        net = resnet18_cbn()
    elif netname== 'resnet50_cbn':
        from models.resnet_nobn import resnet50_cbn
        net = resnet50_cbn()
    elif netname== 'resnet50_nobn':
        from models.resnet_nobn import resnet50_nobn
        net = resnet50_nobn()
    elif netname== 'resnet101_cbn':
        from models.resnet_nobn import resnet101_cbn
        net = resnet101_cbn()
    elif netname== 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif netname== 'densenet121_cbn':
        from models.densenet_nobn import densenet121
        net = densenet121()
    elif netname== 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif netname== 'shufflenetv2_cbn':
        from models.shufflenetv2_nobn import shufflenetv2_cbn
        net = shufflenetv2_cbn()
    elif netname== 'shufflenetv2_nobn':
        from models.shufflenetv2_nobn import shufflenetv2_nobn
        net = shufflenetv2_nobn()
    elif netname== 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif netname== 'squeezenet_nobn':
        from models.squeezenet_nobn import squeezenet_nobn
        net = squeezenet_nobn()
    elif netname== 'squeezenet_cbn':
        from models.squeezenet_nobn import squeezenet_cbn
        net = squeezenet_cbn()
    elif netname== 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif netname== 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif netname== 'seresnet18_cbn':
        from models.senet_nobn import seresnet18
        net = seresnet18()
    elif netname== 'seresnet50_cbn':
        from models.senet_nobn import seresnet50 
        net = seresnet50()
    elif netname=='fixup_cbn':
        from models.fixup_resnet_cifar import fixup_resnet56
        net = fixup_resnet56(cbn=True)
    elif netname=='fixup':
        from models.fixup_resnet_cifar import fixup_resnet56
        net = fixup_resnet56()
    elif netname=='mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif netname=='mobilenetv2_cbn':
        from models.mobilenetv2_nobn import mobilenetv2
        net = mobilenetv2()
    else:
        print(netname)
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
      #  net = torch.nn.parallel.DataParallel(net)
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, num_workers=num_workers, batch_size=batch_size,shuffle=shuffle)        
    return cifar100_training_loader
def get_training_dataloader10(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar10_training = CIFAR100Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_parttrain_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    
    from torch.utils.data import Dataset
    class Subset(Dataset):
        """
        Subset of a dataset at specified indices.

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
        """
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)    

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    num_train = len(cifar100_training)
    indices = list(range(num_train))
    split = int(num_train*3/4)
    np.random.shuffle(indices)
    
    part_train_idx=indices[:split]
    part_test_idx=indices[split:]
    train_subdataset=Subset(cifar100_training,part_train_idx) 
    test_subdataset=Subset(cifar100_training,part_test_idx)
    
    cifar100_parttrain_loader = DataLoader(
        train_subdataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_parttest_loader = DataLoader(
        test_subdataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_parttrain_loader,cifar100_parttest_loader
    
def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader
def get_test_dataloader10(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [1e-4 + base_lr * self.last_epoch / (self.total_iters + 1e-6) for base_lr in self.base_lrs]
