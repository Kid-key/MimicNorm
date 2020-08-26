

import os
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from utils import *
import math

#seed=112
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
CIFAR_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TIME_NOW = datetime.now().isoformat()
CHECKPOINT_PATH = 'checkpoint' 
ENDEPOCH=150


def meanweigh(module):
 #   for name, module in container.named_modules():     
        if isinstance(module, nn.Conv2d) and module.in_channels*module.kernel_size[0] * module.kernel_size[1]/module.groups>50:  
            datavalue=module.weight.data
            meanvalue=datavalue.mean([1,2,3],True)
            module.weight.data=(datavalue-meanvalue)
     #   elif isinstance(module, nn.Linear):  
      #      datavalue=module.weight.data
       #     meanvalue=datavalue.mean([1],True)
        #    module.weight.data=(datavalue-meanvalue)


def getnames_dconv(container):
    
    list0 =[]
    for name, module in container.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels*module.kernel_size[0] * module.kernel_size[1]/module.groups<50:
            list0.append(name+'.weight')
            print((name,module.in_channels*module.kernel_size[0] * module.kernel_size[1]/module.groups))
           # module.weight.requires_grad=False
   # print(list0)
    return list0


def train(epoch,net,optimizer,warmup_scheduler):
    trainloss=0.0
    correct1=0.0
    net.train()

    for batch_index, (images, labels) in enumerate(cifar_training_loader):


      #  images = Variable(images)
      #  labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        _, preds = outputs.max(1)
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        outputs = net(images)
        loss = loss_function(outputs, labels)
        _, preds = outputs.max(1)

        last_layer = list(net.children())[-1]
        correct1 += preds.eq(labels).sum()

        trainloss+=loss.item()*labels.shape[0]
        if 'mean' in args.weight:
            net.apply(meanweigh)
        if epoch < args.warm:
            warmup_scheduler.step()
    total_samples=len(cifar_training_loader.dataset)
    loss=trainloss/total_samples
    acc=correct1.float() /total_samples
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\t acc: {:0.6f}\t LR: {:0.6f}'.format(
        loss,acc,optimizer.param_groups[0]['lr'],epoch=epoch) )
    return acc,loss

def eval_training(epoch,net):
    #net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0

   # net.apply(reset_test)

    for (images, labels) in cifar_test_loader:

        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()*labels.shape[0]
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
    acc=correct.float() / len(cifar_test_loader.dataset)
    loss=test_loss/ len(cifar_test_loader.dataset)
    print('Test Loss and Accuracy:%2d - %.6f %.4f' % (epoch,loss,acc))
    return acc,loss

def inspect_bn(m):
    if isinstance(m, nn.BatchNorm1d):
       # m.train()
        print(m.running_var.data[:10])

if __name__ == '__main__':
   # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--arch', type=str, required=True, help='net type')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-ds', type=str, default = 'cifar100', help='dataset type')
    parser.add_argument('--gpu', default = '0',help='GPU id')
    parser.add_argument('--warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('--resume', type=str, default='', help='checkpoint resume')
    parser.add_argument('--weight', type=str, default='none', help='choices from {\'mean\', \'none\'}, whether to do weight mean or not')
    parser.add_argument('-f','--logfile', type=str, default='std.out')
    parser.add_argument('--remark', type=str, default='')
    args = parser.parse_args()
    print(args)

    if (not args.logfile==''):
        f = open(args.logfile,'a')
    
    if args.gpu=='-1':
        use_gpu=False
        print(use_gpu)
    else:
        use_gpu=True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.ones(1).cuda()
    net = get_network(args.arch, use_gpu=use_gpu)

    if 'mean' in args.weight:
        net.apply(meanweigh)

    #data preprocessing:
    if '100' in args.ds:
        get_training_ds=get_training_dataloader
        get_test_ds=get_test_dataloader
    else:
        get_training_ds=get_training_dataloader10
        get_test_ds=get_test_dataloader10
    cifar_training_loader = get_training_ds(
        CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD,num_workers=4,
        batch_size=args.b,shuffle=True)

    cifar_test_loader = get_test_ds(
        CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD,num_workers=4,
        batch_size=args.b,shuffle=False)

#    cifar_test_loader = cifar_training_loader
   # cifar_training_loader,cifar_test_loader=get_parttrain_dataloader(
    #    CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD,num_workers=4,
    #    batch_size=args.b,shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    #listdconv=getnames_dconv(net)
    #parameters_dconv = [p[1] for p in net.named_parameters() if p[0] in listdconv]
    #parameters_normal = [p[1] for p in net.named_parameters() if not (p[0] in listdconv)]
    #assert False
    #optimizer = optim.SGD([{'params': parameters_normal, 'lr': args.lr}, {'params': parameters_dconv, 'lr': args.lr/10}], momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
 #   train_scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
#    train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=0.15)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,110,135], gamma=0.1) #learning rate decay
    iter_per_epoch = len(cifar_training_loader)
    #getnames_dconv(net)
    warmup_scheduler = None
    if args.warm>0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.arch, TIME_NOW)
    print(TIME_NOW)

    if (args.remark=='') :
        args.remark=str(TIME_NOW)

    if (not args.logfile==''):
        f.write(args.remark+': \n')

    epoch0=0
    state_dict=net.state_dict()
    k0=0

    if args.resume!='' and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        name_par=list(checkpoint.keys())
        print("=> loaded checkpoint '%s' ", args.resume)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        parstr=args.resume.split('-')
        epoch0=int(parstr[-2])+1
    best_acc = 0.0
    lastepoch=0
    for epoch in range(epoch0, ENDEPOCH):

        acctrain,losstrain =train(epoch,net,optimizer,warmup_scheduler)
      #  if (not args.logfile==''):
        #    f.write('%.4f %.4f' %(acctrain,losstrain)+' ')
        #writer.add_scalar("train/loss", losstrain,epoch)
        #writer.add_scalar("train/acc", acctrain,epoch)
        #if epoch==45:
         #   net.apply(freeze_bn)
        net.eval()
        acc,loss = eval_training(epoch,net)
        if epoch >= args.warm:
            train_scheduler.step()
        #writer.add_scalar("test/loss", loss,epoch)
        #writer.add_scalar("test/acc", acc,epoch)
        if (not args.logfile==''):
            f.write('%.4f' %acc+',')
        if epoch==1:
           os.system('nvidia-smi --format=csv,noheader --query-compute-apps=used_gpu_memory -i '+args.gpu)
        if epoch==-40:
            torch.save({'model':net.state_dict(), 
                        'optimizer':optimizer.state_dict()},args.arch+'_'+str(epoch)+'.pth')
        if acc>best_acc:
            best_acc=acc
            best_epoch=epoch
        net.apply(inspect_bn)
    print('best %2d - %.4f' % (best_epoch,best_acc))
    f.write('best %2d - %.4f' % (best_epoch,best_acc)+'\n')
    print(datetime.now().isoformat())
    #writer.close()
    if (not args.logfile=='') and os.path.exists(args.logfile):
        f.write('\n')
        f.close()
