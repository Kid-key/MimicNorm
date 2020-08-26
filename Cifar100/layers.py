import torch.autograd
import torch
import torch.nn as nn
import numpy as np

MODE='none'
Affine=True
print(MODE)
print((' true set affine',Affine))

class MyPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, dZ):
        return dZ*1.0

class MyPassLayer(nn.Module):
    def __init__(self,channelnum,affine=False):
        super(MyPassLayer, self).__init__()
        self.affine=affine
        if self.affine:
            self.mybias = nn.Parameter(torch.Tensor(1,channelnum,1,1))
        else:
            self.register_parameter('mybias', None)
        #self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.mybias)
    def forward(self, x):
        if self.affine:
            out=MyPass.apply(x)+self.mybias
        else:
            out=MyPass.apply(x)
        return out

class MyScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,scale):
        ctx.scale=scale
        return input*scale
    @staticmethod
    def backward(ctx, dZ):
        return dZ*ctx.scale,dZ.mean()

class MyScaleLayer(nn.Module):
    def __init__(self,initvalue=0):
        super(MyScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.tensor(initvalue))
        #self.scale.data.fill_(initvalue)
    def forward(self, x):
        out=MyScale.apply(x,self.scale)
        return out

class MyMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,beta):
        size = input.size()
        size_prods = size[0]
        sumdim=[0]
        start=2
        for i in range(start,len(size)):
            size_prods *= size[i]
            sumdim+=[i]
        mean=input.sum(sumdim,True)/(size_prods)
        Var = (torch.sqrt(((input-mean)**2).sum(sumdim,True)/(size_prods))+1e-5)
        ctx.sumdim=sumdim
        ctx.beta=beta
        ctx.size_prods=size_prods
        inputnorm=  input-mean
        if not beta is None:
            inputnorm=  inputnorm+beta

        returndata=(inputnorm,mean,Var)
        return returndata

    @staticmethod
    def backward(ctx, dZ,null3,null4):
        grad_input=dZ
        dBeta = torch.sum(dZ, ctx.sumdim,True)

        grad_input = (dZ - dBeta /ctx.size_prods )
        if ctx.beta is None:
            dBeta=None
        return grad_input,dBeta

class MyBatchMeanLayer(nn.Module):
    def __init__(self, channelnum,momentum=0.1,affine=Affine):
        super(MyBatchMeanLayer, self).__init__()
        self.affine=affine
        self.momentum=momentum
        if self.affine:
            self.mybias = nn.Parameter(torch.Tensor(1,channelnum,1,1))
        else:
            self.register_parameter('mybias', None)
        self.register_buffer('myrun_mean', torch.zeros(1,channelnum,1,1))
        self.register_buffer('myrun_var', torch.zeros(1,channelnum,1,1))
        self.reset_parameters()

    def reset_running_stats(self):
        self.myrun_mean.zero_()
        self.myrun_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.mybias)

    def forward(self, input):

        if self.training:
            output,mean,var=MyMean.apply(input, self.mybias)
            self.myrun_mean.data=self.myrun_mean.data*(1-self.momentum)+self.momentum*mean
            self.myrun_var.data=self.myrun_var.data*(1-self.momentum)+self.momentum*var
        else:
            output=(input-self.myrun_mean)
            if self.affine:
                output=output+ self.mybias

        # print(invscale.mean().detach().cpu().numpy(),invscale.std().detach().cpu().numpy())
        return output
        # return MyMean.apply(input,self.weight,self.mybias)

class MyNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,gamma,beta):
        BN=False
        size = input.size()
        size_prods = size[0]
        sumdim=[0]
        start=2
        for i in range(start,len(size)):
            size_prods *= size[i]
            sumdim+=[i]
        mean=input.sum(sumdim,True)/(size_prods)
        Var = (torch.sqrt(((input-mean)**2).sum(sumdim,True)/(size_prods)+1e-5))
        ctx.save_for_backward(input)
        ctx.sumdim=sumdim
        ctx.gamma=gamma
        ctx.size_prods=size_prods
        if gamma is None:
            inputnorm=  (input-mean)/Var
        else:
            inputnorm=  (input-mean)/Var*gamma+beta

        returndata=(inputnorm,mean,Var)
        return returndata

    @staticmethod
    def backward(ctx, dZ,null3,null4):
        grad_input=dZ
        input, = ctx.saved_tensors
        mean=input.sum(ctx.sumdim,True)/(ctx.size_prods)
        InvVar = 1/(torch.sqrt(((input-mean)**2).sum(ctx.sumdim,True)/(ctx.size_prods)+1e-5))
        Xnorm=  InvVar*(input-mean)
        dBeta = torch.sum(dZ, ctx.sumdim,True)
        dGamma = torch.sum(dZ * Xnorm, ctx.sumdim,True)
        grad_input = (dZ - dBeta /ctx.size_prods - dGamma * Xnorm /ctx.size_prods)* InvVar

        if ctx.gamma is None:
            dGamma=None
            dBeta=None
        else:
            grad_input=grad_input*ctx.gamma

        return grad_input,dGamma,dBeta

class MyBatchNormLayer(nn.Module):
    def __init__(self, channelnum,momentum=0.1,affine=Affine):
        super(MyBatchNormLayer, self).__init__()
        self.affine=affine
        self.momentum=momentum
        if self.affine:
            self.myweight = nn.Parameter(torch.Tensor(1,channelnum,1,1))
            self.mybias = nn.Parameter(torch.Tensor(1,channelnum,1,1))
        else:
            self.register_parameter('myweight', None)
            self.register_parameter('mybias', None)
        self.register_buffer('myrun_mean', torch.zeros(1,channelnum,1,1))
        self.register_buffer('myrun_var', torch.ones(1,channelnum,1,1))
        self.register_buffer('test_mean', torch.zeros(1,channelnum,1,1))
        self.register_buffer('test_var2', torch.ones(1,channelnum,1,1))
        self.register_buffer('temp_number', torch.zeros(1))
        self.reset_parameters()

    def reset_running_stats(self):
        self.myrun_mean.zero_()
        self.myrun_var.fill_(1)
    def reset_test_stats(self):
        self.test_mean.zero_()
        self.test_var2.fill_(1)
        self.temp_number.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        self.reset_test_stats()
        if self.affine:
            nn.init.ones_(self.myweight)
            nn.init.zeros_(self.mybias)

    def forward(self, input):

        if self.training:
            output,mean,var=MyNorm.apply(input,self.myweight, self.mybias)
            self.myrun_mean.data=self.myrun_mean.data*(1-self.momentum)+self.momentum*mean
            self.myrun_var.data=self.myrun_var.data*(1-self.momentum)+self.momentum*var
            # self.num_batches_tracked.data=(output>0).sum([0,2,3]).float()*output.size(1)/output.numel()
        else:
            output=(input-self.myrun_mean)/self.myrun_var
            size = output.size()
            size_prods = size[0]
            sumdim=[0]
            start=2
            for i in range(start,len(size)):
                size_prods *= size[i]
                sumdim+=[i]
            Mean=output.sum(sumdim,True)/(size_prods)
            Var2 = (((output-Mean)**2).sum(sumdim,True)/(size_prods))
            self.test_mean.data=(self.test_mean.data*self.temp_number.data+Mean*size[0])/(self.temp_number.data+size[0])
            self.test_var2.data=(self.test_var2.data*self.temp_number.data+Var2*size[0])/(self.temp_number.data+size[0])
            self.temp_number.data=self.temp_number.data+size[0]
            if self.affine:
                output=output*self.myweight+ self.mybias

        # print(invscale.mean().detach().cpu().numpy(),invscale.std().detach().cpu().numpy())
        return output

if MODE=='mean':
    Layer=MyBatchMeanLayer
elif MODE=='norm':
    Layer=MyBatchNormLayer
#    Layer=nn.BatchNorm2d
else:
    Layer=MyPassLayer
