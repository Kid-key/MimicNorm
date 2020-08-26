import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
from datetime import datetime

CHECKPOINT_PATH = 'checkpoint'


TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'
CHECKPOINT_PATH = 'checkpoint'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# from tensorboardX import SummaryWriter
import math

# from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy



def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def linear_learning_rate(optimizer, epoch, init_lr,T_max=100):
    lr = 2e-5 - init_lr/T_max*(epoch+1-T_max)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#def consine_learning_rate(optimizer, epoch, init_lr):
 #   """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  #  lr = 2e-5 + init_lr*(1+math.cos(math.pi*epoch/T_max))/2
   # for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
# import imagenet_seq
# ImagenetLoader=imagenet_seq.data.Loader

model_names = [
    'vgg11_cbn','vgg11_bn', 'vgg16_cbn', 'vgg16_bn',
    'resnet18', 'resnet50', 'resnet101', 'resnet18_cbn', 'resnet50_cbn', 'resnet101_cbn', 'shufflenetv2', 'shufflenetv2_cbn'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a','--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
#parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                 #   help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print_freq', '-f', default=500, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-mark', type=str, default='', help='remark to this experiment')
parser.add_argument('--warm', type=int, default=2, help='warm up training phase')
parser.add_argument('-record', type=bool, default=True, help='whether to save checkpoint and events')
parser.add_argument('--gpu', default = '0',help='GPU id')
parser.add_argument('--weight', default = 'none',help='none or mean')
args = parser.parse_args()
print(TIME_NOW)
from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (0.01+self.last_epoch / (self.total_iters + 1e-8)) for base_lr in self.base_lrs]

#from models.squeezenet import squeezenet1_0, squeezenet1_1

#from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.vgg import vgg11_cbn, vgg11_bn,vgg16_cbn, vgg16_bn

from models.resnet_nobn import resnet18_cbn, resnet50_cbn, resnet101_cbn
from models.resnet_true import resnet18,  resnet50, resnet101
     

def meanweigh(module):
 #   for name, module in container.named_modules():     
        if isinstance(module, nn.Conv2d) and module.groups==1:  
            datavalue=module.weight.data
            meanvalue=datavalue.mean([1,2,3],True)
            module.weight.data=(datavalue-meanvalue)
     #   elif isinstance(module, nn.Linear):  
      #      datavalue=module.weight.data
       #     meanvalue=datavalue.mean([1],True)
        #    module.weight.data=(datavalue-meanvalue)
    
best_prec1 = 0.0
def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
     

    # switch to train mode
    model.train()
    #model.apply(freeze_bn)
    dslen=len(train_loader)
    end = time.time()
    for k,(input, target) in enumerate(train_loader):
        if epoch < args.warm:
            warmup_scheduler.step()


        # measure data loading time
        if True:
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec= accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec[0].item(), input.size(0))
            top5.update(prec[1].item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if 'mean' in args.weight:
            model.apply(meanweigh)#(model)
            
        n_iter = (epoch) * len(train_loader) + k 
        if n_iter==1:
            os.system('nvidia-smi')
        if n_iter%print_freq==1 and epoch<args.start_epoch+5:
            print('Epoch: [{0}][{1}/{2}]\t'
                   'LR: {3:.5f}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'LossEpoch {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.3f}\t'
                      'Prec@5 {top5.avg:.3f}\t'.format(
                    epoch, n_iter,dslen, optimizer.param_groups[0]['lr'],batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()
            top5.reset()
            #validatetrain(val_loader, model, criterion)
        elif k==dslen-1:
            model.apply(inspect_bn)
            print('Epoch: [{0}][{1}/{2}]\t'
                   'LR: {3:.5f}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'LossEpoch {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.3f}\t'
                      'Prec@5 {top5.avg:.3f}\t'.format(
                    epoch, n_iter,dslen, optimizer.param_groups[0]['lr'],batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)

                      
            
           
def freeze_bn(m):
    if isinstance(m, nn.BatchNorm1d):
       # m.train()
        m.track_running_stats=False

def inspect_bn(m):
    if isinstance(m, nn.BatchNorm1d):
       # m.train()
        print(m.running_var.data[:10])

#model.apply(freeze_bn)
def validatetrain(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec= accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec[0].item(), input.size(0))
            top5.update(prec[1].item(), input.size(0))
    model.train()
    print(' *TRAINMODE Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

def validate(val_loader, model, criterion, print_freq,epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
   # model.apply(unfreeze_bn)

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec= accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec[0].item(), input.size(0))
            top5.update(prec[1].item(), input.size(0))

            # measure elapsed time

            if i % print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch,
                    i, len(val_loader),  loss=losses,
                    top1=top1, top5=top5))
        n_iter = (epoch) * len(train_loader) + i + 1 
                
        # if args.record:
        #     writer.add_scalar('Test/Average loss', losses.val,  n_iter)
        #     writer.add_scalar('Test/Accuracy', top1.val,  n_iter)

    model.train()
    f_loss.write('\n epoch {} test with loss {:.4f} \n'.format(epoch,losses.avg)) 
    f_acc.write('\n epoch {} test with top1 {:.3f} and top5 {:.3f} \n'.format(epoch,top1.avg,top5.avg))  
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    

    return top1.avg, top5.avg

if args.gpu!='-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
if args.mark=='':
    args.mark=TIME_NOW
        
checkpoint_path = os.path.join(CHECKPOINT_PATH,args.mark)
runslogdir=os.path.join(LOG_DIR, args.arch, args.mark)
if not os.path.exists(runslogdir):
    os.makedirs(runslogdir)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

f_loss = open(os.path.join(runslogdir,'loss.txt'),'a') 
f_acc = open(os.path.join(runslogdir,'acc.txt'),'a') 


if __name__ == '__main__':


        
        
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))


    if args.arch == 'vgg11_cbn':
        model = vgg11_cbn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_cbn':
        model = vgg16_cbn()
    elif args.arch == 'vgg19_cbn':
        model = vgg19_cbn()
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn()
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn()
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet18_cbn':
        model = resnet18_cbn()
    elif args.arch == 'resnet50_cbn':
        model = resnet50_cbn()
    elif args.arch == 'resnet101_cbn':
        model = resnet101_cbn()
    elif args.arch == 'shufflenetv2':
        from models.shufflenetv2 import shufflenet_v2_x0_5
        model = shufflenet_v2_x0_5(pretrained=args.pretrained)
    elif args.arch == 'shufflenetv2_cbn':
        from models.shufflenetv2_cbn import shufflenet_v2_x0_5
        model = shufflenet_v2_x0_5()
    else:
        raise NotImplementedError

    # use cuda

    if ',' in args.gpu:
        #torch.distributed.init_process_group(backend='nccl', 
                   #             world_size=2, rank=0)
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.parallel.DataParallel(model)
    model.cuda()

    # Data loading
    train_loader, val_loader = data_loader('', args.batch_size, args.workers)
    # train_loader = ImagenetLoader(args.data+'/train', batch_size=args.batch_size, num_workers=args.workers)
    # train_loader=val_loader

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr * args.batch_size / 256.,
                          momentum=args.momentum,nesterov=False, 
                          weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,85], gamma=0.1) #learning rate decay
    #train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=5e-4, T_max=args.epochs) #learning rate decay


    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'para' in checkpoint.keys():
                optstate=checkpoint['opt']
                checkpoint=checkpoint['para']
                optimizer.load_state_dict(optstate)
                print("=> loaded optimizer from '%s' ", args.resume)
            try:
                model.load_state_dict(checkpoint)
                #print(optimizer.param_groups[0]['lr'])                
            except:
                state_dict=model.state_dict()
                name_par=list(checkpoint.keys())
                k0=0
                for key,para in state_dict.items():
                    if k0<len(name_par) and para.numel()==checkpoint[name_par[k0]].numel():
                        state_dict[key]=checkpoint[name_par[k0]]
                        k0+=1  
                model.load_state_dict(state_dict)
            parstr=args.resume.split('-')
            if args.start_epoch==0:
                args.start_epoch =int(parstr[-2])+1
            #import math
            #optimizer.param_groups[0]['lr']*=1/2*(1+math.cos(math.pi*args.start_epoch/130.0))
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, int(parstr[-2])))
            #warmup_scheduler = WarmUpLR(optimizer, 0)
    else:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch*args.warm )
        print("=> no checkpoint found at '{}'".format(args.resume))


    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False
    
    #torch.backends.cudnn.enabled = True

    if 'mean' in args.weight:
        model.apply(meanweigh)#(model)
    

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq,0)
        assert False
        
    lastepoch=0

    for epoch in range(args.start_epoch,args.epochs):
        #print(optimizer.param_groups[0]['lr'])
        if epoch >= args.warm:
            #linear_learning_rate(optimizer, epoch, args.lr,T_max=args.epochs)
            train_scheduler.step(epoch)
            #print((epoch,optimizer.param_groups[0]['lr']))
            #continue
        # train for one epoch
        try:
            train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
            prec1, prec5 = validate(val_loader, model, criterion, args.print_freq,epoch)
        except Exception as e:
            print(e)
            train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers)
            print('DataLoader Reset')
            #assert False
            continue
        else:
            # remember the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if epoch > 55 and is_best:
                # if lastepoch>0:
                #     os.remove(checkpoint_path.format(net=args.arch, epoch=lastepoch, type='best'))
                torch.save(model.state_dict(), checkpoint_path.format(net=args.arch, epoch=epoch, type='best'))
                lastepoch=epoch
            if epoch%6==5:
                torch.save({'para':model.state_dict(),'opt':optimizer.state_dict()}, checkpoint_path.format(net=args.arch, epoch=epoch, type='regular'))
    f_loss.close()
    f_acc.close()
    print('Best Accuracy is {acc} in {epoch}'.format(acc=best_prec1,epoch=lastepoch))


