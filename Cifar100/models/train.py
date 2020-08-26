# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader,WarmUpLR,get_parttrain_dataloader
# get_test_dataloader=get_parttrain_dataloader
import math


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

def meanweigh(container,statdict,prefix):
    # Iterate through model, insert quantization functions as appropriate
    if prefix:
        prefix0=prefix+'.'
    else:
        prefix0=prefix
    for name, module in container.named_children():     
        if isinstance(module, nn.Conv2d): 
            datavalue=module.weight.data
            # size=datavalue.size()
            # fixvar=math.sqrt(2.0/(size[1]*size[2]*size[3]))
            meanvalue=datavalue.mean([1,2,3],True)
            # print(meanvalue.abs().mean([0]).cpu().numpy(),datavalue.abs().mean([0,1,2,3]).cpu().numpy())
            # varvalue = (torch.sqrt(((datavalue-meanvalue)**2).mean([1,2,3],True))+1e-5) 
            # print(prefix0+name+'.weight')
            # print(varvalue.view(-1).mean(),fixvar)
            statdict[prefix0+name+'.weight']=(datavalue-meanvalue)#/varvalue*fixvar
        # elif isinstance(module, nn.Linear):            
        #     statdict[prefix0+name+'.weight']=module.weight.data-module.weight.data.mean([1],True)
        if has_children(module):
            meanweigh(module,statdict,prefix0+name)
    return statdict
    
def meanweighgrad(container,prefix=''):
    # Iterate through model, insert quantization functions as appropriate
    for name, module in container.named_children():     
        if isinstance(module, nn.Conv2d): 
            module.weight.grad-=module.weight.grad.mean([1,2,3],True)    
        # elif isinstance(module, nn.Linear):
        #     module.weight.grad-=module.weight.grad.mean([1],True)    
        if has_children(module):
            if prefix:
                prefix0=prefix+'.'+name
            else:
                prefix0=name
            meanweighgrad(module,prefix0)
class outfunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs):
        # ctx is a context object that can be used to stash information
        # for backward computation
        outtop5, _ = outputs.topk(5, 1, True, True)
        # outputs-=outtop5[:,-1].unsqueeze(1)
        # outputs*=(outputs>0).float()
        scale,_=(outtop5[:,1:]-outtop5[:,:-1]).max(1)
        outputs=outputs/scale.unsqueeze(1)*5
        return outputs 

    @staticmethod
    def backward(ctx, grad_output):    
        return grad_output , None
def train(epoch,use_gpu=True):

    net.train()
    trainloss=0.0
    correct1=0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        # if epoch <= args.warm:
        #     warmup_scheduler.step()


        images = Variable(images)
        labels = Variable(labels)

        if use_gpu:
            labels = labels.cuda()
            images = images.cuda() 
                 
        for i in range(1):
            optimizer.zero_grad()
            
            outputs = net(images)
            # outtop5, _ = outputs.topk(5, 1, True, True)
            # scale,_=(outtop5[:,1:]-outtop5[:,:-1]).max(1)
            # outputs=outputs/scale.unsqueeze(1)
            # outputs=outfunc.apply(outputs)
            # print(outputs.mean(0),outputs.std(0))
            # outmean=outputs.sum(0,True).expand_as(outputs)
            loss = loss_function(outputs, labels) #+((outputs-outmean)**2/args.b-1).norm()
            _, preds = outputs.max(1)
            loss.backward()
            # meanweighgrad(net)
            optimizer.step()
            
            if args.meanweight:
                state_dict=net.state_dict()
                # print(state_dict.keys())
                Newstate_dict=meanweigh(net,state_dict,'')
                net.load_state_dict(Newstate_dict)

            n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
            
            # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tacc: {:0.4f}\tLR: {:0.6f}'.format(
            #         preds.eq(labels).sum(),
            #         optimizer.param_groups[0]['lr'],
            #         epoch=i,
            #         trained_samples=batch_index * args.b + len(images),
            #         total_samples=len(cifar100_training_loader.dataset)
            #     ))
            # if args.record:                     
            #     for name, param in net.named_parameters():
            # 
            #         layer, attr = os.path.splitext(name)
            #         attr = attr[1:]
            #         writer.add_histogram("{}/{}".format(layer, attr), param, i)       
            #         writer.add_histogram("{}/{}_grad".format(layer, attr), param.grad, i)     

                
        correct1 += preds.eq(labels).sum()
        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_buffers():
        #     if 'invscale' in name:
        #         # print(para.data)
        #         writer.add_scalar('LastLayer/InvVar', para.data, n_iter)
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        # if batch_index%100==0:
        #     print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #         loss.item(),
        #         optimizer.param_groups[0]['lr'],
        #         epoch=epoch,
        #         trained_samples=batch_index * args.b + len(images),
        #         total_samples=len(cifar100_training_loader.dataset)
        #     ))

        #update training loss for each iteration
        trainloss+=loss.item()
        
    total_samples=len(cifar100_training_loader.dataset) 
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\t acc: {:0.6f}\t LR: {:0.6f}'.format(
                trainloss/total_samples,
                correct1.float() /total_samples,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
            )) 
    if args.record:       
        writer.add_scalar('Train/Average loss', trainloss/total_samples, epoch)
        writer.add_scalar('Train/Accuracy', correct1.float() /total_samples, epoch)
        
        for name, param in net.named_parameters():

            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch,use_gpu=True):
    net.eval()

    test_loss = 0.0 # cost function error
    correct1 = 0.0
    correct5 = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)        
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct1 += preds.eq(labels).sum()
        
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct5 += pred.eq(labels.view(1, -1).expand_as(pred)).view(-1).float().sum(0)


    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, top5: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct1.float() / len(cifar100_test_loader.dataset),
        correct5/ len(cifar100_test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    if args.record:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct1.float() / len(cifar100_test_loader.dataset), epoch)

    return correct1.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default = 'vgg11',help='net type')
    parser.add_argument('-gpu', default = '1',help='GPU id')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-gamma', type=float, default=0.2, help='lr drop down')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', type=str, default='', help='checkpoint resume')
    parser.add_argument('-mark', type=str, default='', help='remark to this experiment')
    parser.add_argument('--norecord', dest='record', action='store_false', help='whether to save checkpoint and events')
    parser.add_argument('--record', dest='record', action='store_true', help='whether to save checkpoint and events')
    parser.add_argument('--nomean', dest='meanweight', action='store_false', help='whether to')
    parser.set_defaults(record=False)
    parser.set_defaults(meanweight=True)
    args = parser.parse_args()
    print(args)
    print(settings.TIME_NOW)
    if args.gpu=='-1':
        use_gpu=False
        print(use_gpu)
    else:
        use_gpu=True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    net = get_network(args.net,use_gpu=use_gpu)
        
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,90,120,150], gamma=args.gamma) #learning rate decay
    # iter_per_epoch = len(cifar100_training_loader)
    # train_scheduler=optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH,settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.mark=='':
        args.mark=settings.TIME_NOW
    else:
        args.record=True
    if args.record:
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, args.mark))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    # if use_gpu:
    #     input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    # else:
    #     input_tensor = torch.Tensor(12, 3, 32, 32)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))


    best_acc = 0.0
    epoch0=1
    state=1
    if args.resume!='' and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        print("=> loaded checkpoint '%s' ", args.resume)
        net.load_state_dict(checkpoint)
        parstr=args.resume.split('-')
        epoch0=int(parstr[1])
    
    lastepoch=0
    for epoch in range(epoch0, 180):
        # if epoch > 50:
        #     train_scheduler.step(epoch-50)
        train_scheduler.step(epoch)

        train(epoch,use_gpu)
        acc = eval_training(epoch,use_gpu)
        
        # if acc>0.3 and state==0:
        #     state_dict=net.state_dict()
        #     print(state_dict.keys())
        #     torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='trun1'))
        #     net=get_network(args.net+'_nobn',use_gpu=use_gpu)
        #     state_dict0 = net.state_dict()
        #     name_par=list(state_dict0.keys())
        #     k0=0
        #     scale=1
        #     for key,para in state_dict.items():
        #         if k0<len(name_par) and para.numel()==state_dict0[name_par[k0]].numel():
        #             state_dict0[name_par[k0]]=para
        #             lastweight=name_par[k0]
        #             k0+=1
        #         if  'running_var' in key:
        #             print(lastweight)
        #             print([state_dict0[lastweight].size(),para.size()])
        #             state_dict0[lastweight]/=para.mean()             
        #     net.load_state_dict(state_dict0)
        #     state=1  
        #     optimizer = optim.SGD(net.parameters(), lr=args.lr/scale, momentum=0.9, weight_decay=5e-4)
        #     train_scheduler=optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
        #     print('change network at epoch {epoch}'.format(epoch=epoch))      
        # 
        # if epoch > 80 and state==1:
        #     state_dict=net.state_dict()
        #     torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='trun2'))
        #     net=get_network(args.net,use_gpu=use_gpu)
        #     state_dict0 = net.state_dict()
        #     name_par=list(state_dict.keys())
        #     k0=0
        #     for key,para in state_dict0.items():
        #         if para.numel()==state_dict[name_par[k0]].numel():
        #             state_dict0[key]=state_dict[name_par[k0]]
        #             k0+=1  
        #     net.load_state_dict(state_dict0)
        #     state=2       
        #     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        #     train_scheduler=optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)    
        #     print('change network again at epoch {epoch}'.format(epoch=epoch))              

        # start to save best performance model after learning rate decay to 0.01 
        if epoch > 80 and best_acc < acc:
            if args.record:
                if lastepoch>0:
                    os.remove(checkpoint_path.format(net=args.net, epoch=lastepoch, type='best'))
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            lastepoch=epoch
            best_acc = acc
        
        if epoch in [50,100] and args.record:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    if args.record:
        writer.close()
    print('Best Accuracy is {acc} in {epoch}'.format(acc=best_acc,epoch=lastepoch))
