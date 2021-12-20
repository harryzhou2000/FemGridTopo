import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as visMod
import time, os

def caseStamp():
    return '%d_%d' % (round(time.time()), os.getpid())

def modelNParam(model) -> int:
    return sum([param.nelement() for param in model.parameters()])

class gGABlockA(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.convA = nn.Conv2d(channel_in, channel_out,
                               3, padding='same', padding_mode='reflect')
        self.convB1 = nn.Conv2d(channel_in, channel_out*3,
                                3, padding='same', padding_mode='reflect')
        self.convB2 = nn.Conv2d(
            channel_out*3, channel_out*1, 3, padding='same', padding_mode='reflect')
        self.poolA = nn.AvgPool2d(3, padding=1, stride=1)
        self.poolConv = nn.Conv2d(channel_in, channel_out, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.act(self.convA(x) +
                     self.convB2(self.act(self.convB1(x))) + 
                     self.poolConv(self.poolA(x)))
        return y


class gradientGenA(nn.Module):
    def __init__(self, firstGen=(4,4), seq0 = [16,32,16]):
        super().__init__()
        self.bndC2 = nn.Conv2d(3, firstGen[0], 2)
        self.rhoC1 = nn.Conv2d(1, firstGen[1], 1)
        self.Bseq0 = nn.Sequential()
        ilayer = 0
        lastend = firstGen[0]+firstGen[1]
        for layerSz in seq0:
            self.Bseq0.add_module('BLK%d' % (ilayer), gGABlockA(
                    lastend, layerSz))
            lastend = layerSz
            ilayer += 1

        self.out = nn.Conv2d(lastend,1,1)
        
    def forward(self, bnd, rho):
        bnd2 = self.bndC2(bnd)
        rho1 = self.rhoC1(rho)
        dcat = torch.cat((bnd2,rho1), 1)
        dcatS = self.Bseq0(dcat)
        grad = torch.tanh(self.out(dcatS))

        return grad


def runOneEpoch(model, dataset, iepoch, optimizer=None, scheduler=None,  iftrain=False, see=10):
    if iftrain:
        model.train()
    else:
        model.eval()

    lossHist = []

    ib = 0

    for batch in dataset:
        ib += 1
        result = model(batch['bnd'],batch['rho'])

        if iftrain:
            optimizer.zero_grad()
        maxabsDA = batch['da'].abs().max()
        target = batch['da']/maxabsDA
        loss = F.mse_loss(result, target)
        lossHist.append(float(loss))
        if iftrain:
            loss.backward()
            optimizer.step()
            scheduler.step()


        if(ib % see == 0 and iftrain):
            print('Train Epoch %d :: Batch number %4d, loss [%.6e]' % (
                iepoch, ib, loss))
        

    return (lossHist, torch.mean(torch.tensor(lossHist)))

class RhoBndCat(nn.Module):
    def __init__(self, firstGen = [2,2], act = nn.ReLU()):
        super().__init__()
        self.bndC2 = nn.Conv2d(3, firstGen[0], 2)
        self.rhoC1 = nn.Conv2d(1, firstGen[1], 1)
        self.act = act
    
    def forward(self, bnd, rho):
        bnd2 = self.bndC2(bnd)
        rho1 = self.rhoC1(rho)
        dcat = torch.cat((bnd2, rho1), 1)
        return self.act(dcat)

class ConvDownSampler(nn.Module):
    def __init__(self, channelIn,channelOut, nds=2, act = nn.ReLU()):
        super().__init__()
        self.c = nn.Conv2d(channelIn, channelOut, nds, stride=nds, padding=0)
        self.act = act

    def forward(self, x):
        return self.act(self.c(x))

class ResNetForwardA(nn.Module):
    def __init__(self, channel, midchannel, ksize, act = nn.ReLU()):
        super().__init__()
        self.c1 = nn.Conv2d(channel, midchannel, ksize,
                            padding='same', padding_mode='reflect')
        self.c2 = nn.Conv2d(midchannel, channel, ksize,
                            padding='same', padding_mode='reflect')
        self.act = act

        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        return self.act(self.bn(self.c2(self.act(self.c1(x))) + x))

def PosEncoding2D(shape, maxSize, device):
    if(shape[1] % 4 != 0):
        raise ValueError("Input channel not 4 times")
    M = maxSize **  (4 * torch.arange(shape[1]//4, device=device)/shape[1])
    iH,iW = torch.meshgrid((torch.arange(shape[2],device=device),torch.arange(shape[3],device=device)), indexing='ij')
    Hc = torch.cos(iH.view((1,shape[2],shape[3])) / M.view((shape[1]//4,1,1)))
    Wc = torch.cos(iW.view((1,shape[2],shape[3])) / M.view((shape[1]//4,1,1)))
    Hs = torch.sin(iH.view((1,shape[2],shape[3])) / M.view((shape[1]//4,1,1)))
    Ws = torch.sin(iW.view((1,shape[2],shape[3])) / M.view((shape[1]//4,1,1)))
    E = torch.cat((Hc,Wc,Hs,Ws),dim=0)
    return E.expand(shape[0],-1,-1,-1)

class AddPosEncoding2D(nn.Module):
    def __init__(self,maxSize, ratioInit = 1.0):
        super().__init__()
        self.maxSize = maxSize
        self.ratio = torch.nn.parameter.Parameter(torch.tensor(ratioInit))
    
    def forward(self, x):
        return x + PosEncoding2D(x.size(), self.maxSize, x.device) * self.ratio



class PixelwiseSelfAttention2d(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.att = nn.MultiheadAttention(dim, head, batch_first= True)
        self.bn = nn.BatchNorm2d(dim)
        self.dim = dim

    def forward(self, x, need_weights = False):
        # x : (N,C,H,W)
        N = x.size(0);C = x.size(1);H = x.size(2);W = x.size(3)
        if(C != self.dim):
            raise ValueError("PixelwiseSelfAttention2d: input channel %d needed"%(self.dim))
        xp = torch.permute(x.view((N,C,-1)),(0,2,1))
        ret  = self.att(xp, xp, xp, need_weights=need_weights)
        # print(len(ret), need_weights)
        ret = (ret[0].permute(0, 2, 1).view((N, C, H, W)), ret[1])
        retx = ret[0] + x
        return self.bn(retx)


class PI_ABGenerator(nn.Module):
    def __init__(self, firstGen = [2,2]):
        super().__init__()
        self.catter = RhoBndCat(firstGen)
        mlist = []
        elist = []
        act = nn.Softplus()
        mlist.append(ConvDownSampler(sum(firstGen), 16, 2, act))
        mlist.append(PixelwiseSelfAttention2d(16,1))
        mlist.append(ResNetForwardA(16, 8, 3, act)) #32*32

        mlist.append(ConvDownSampler(16, 32, 2, act))
        mlist.append(PixelwiseSelfAttention2d(32, 1))
        mlist.append(ResNetForwardA(32, 16, 3, act))  # 16*16

        mlist.append(ConvDownSampler(32, 64, 2, act))
        mlist.append(PixelwiseSelfAttention2d(64, 1))
        mlist.append(ResNetForwardA(64, 16, 3, act))  # 8*8

        mlist.append(ConvDownSampler(64, 128, 2, act))
        mlist.append(PixelwiseSelfAttention2d(128, 1))
        mlist.append(ResNetForwardA(128, 16, 3, act))  # 4*4

        elist.append(nn.Linear(128 * 4 * 4, 128 * 4))
        elist.append(nn.BatchNorm1d(128 * 4))
        elist.append(act)
        elist.append(nn.Linear(128 * 4, 128))
        elist.append(nn.BatchNorm1d(128))
        elist.append(act)
        elist.append(nn.Linear(128, 16))
        elist.append(nn.BatchNorm1d(16))
        elist.append(act)
        elist.append(nn.Linear(16, 1))

        self.mseq = nn.Sequential()
        i = 0
        for m in mlist:
            self.mseq.add_module('module_%d'%(i), m)
            i = i + 1
        i = 0
        self.eseq = nn.Sequential()
        for m in elist:
            self.eseq.add_module('module_%d' % (i), m)
            i = i + 1

    def forward(self, bnd, rho):
        mseqout = self.mseq(self.catter(bnd, rho))
        return self.eseq(mseqout.view(mseqout.size(0),-1))


class PI_ABGenertorStatic(nn.Module):
    def __init__(self):
        super().__init__()
        mlist = []
        elist = []
        act = nn.Softplus()
        mlist.append(ConvDownSampler(1, 4, 2, act))
        mlist.append(AddPosEncoding2D(64))
        mlist.append(PixelwiseSelfAttention2d(4,1))  
        # mlist.append(ResNetForwardA(4, 2, 3, act)) 
        mlist.append(ResNetForwardA(4, 2, 3, act)) #32*32

        mlist.append(ConvDownSampler(4, 8, 2, act))
        mlist.append(PixelwiseSelfAttention2d(8, 1))
        # mlist.append(ResNetForwardA(8, 4, 3, act)) 
        mlist.append(ResNetForwardA(8, 4, 3, act))  # 16*16

        mlist.append(ConvDownSampler(8, 16, 2, act))
        mlist.append(PixelwiseSelfAttention2d(16, 1))
        # mlist.append(ResNetForwardA(16, 8, 3, act))
        mlist.append(ResNetForwardA(16, 8, 3, act))  # 8*8

        mlist.append(ConvDownSampler(16, 32, 2, act))
        mlist.append(PixelwiseSelfAttention2d(32, 1))
        # mlist.append(ResNetForwardA(32, 16, 3, act))
        mlist.append(ResNetForwardA(32, 16, 3, act))  # 4*4

        elist.append(nn.Linear(32 * 4 * 4, 32 * 4))
        elist.append(nn.BatchNorm1d(32 * 4))
        elist.append(act)
        elist.append(nn.Linear(32 * 4, 32))
        elist.append(nn.BatchNorm1d(32))
        elist.append(act)
        elist.append(nn.Linear(32, 8))
        elist.append(nn.BatchNorm1d(8))
        elist.append(act)
        elist.append(nn.Linear(8, 1))

        self.mseq = nn.Sequential()
        i = 0
        for m in mlist:
            self.mseq.add_module('module_%d'%(i), m)
            i = i + 1
        i = 0
        self.eseq = nn.Sequential()
        for m in elist:
            self.eseq.add_module('module_%d' % (i), m)
            i = i + 1

    def forward(self, rho):
        mseqout = self.mseq(rho)
        return self.eseq(mseqout.view(mseqout.size(0),-1))


class PI_ABGeneratorV2(nn.Module):
    def __init__(self, firstGen=[2, 2]):
        super().__init__()
        self.catter = RhoBndCat(firstGen)
        mlist = []
        elist = []
        act = nn.ReLU()
        
        mlist.append(PixelwiseSelfAttention2d(sum(firstGen), 1))
        mlist.append(ResNetForwardA(sum(firstGen), 2, 3, act))
        mlist.append(ConvDownSampler(sum(firstGen), 16, 4, act))  # 16*16

        mlist.append(PixelwiseSelfAttention2d(16, 1))
        mlist.append(ResNetForwardA(16, 8, 3, act))
        mlist.append(ConvDownSampler(16, 32, 4, act))  # 4*4

        elist.append(nn.Linear(32 * 4 * 4, 32 * 4))
        elist.append(nn.BatchNorm1d(32 * 4))
        elist.append(act)
        elist.append(nn.Linear(32 * 4, 32))
        elist.append(nn.BatchNorm1d(32))
        elist.append(act)
        elist.append(nn.Linear(32, 1))


        self.mseq = nn.Sequential()
        i = 0
        for m in mlist:
            self.mseq.add_module('module_%d' % (i), m)
            i = i + 1
        i = 0
        self.eseq = nn.Sequential()
        for m in elist:
            self.eseq.add_module('module_%d' % (i), m)
            i = i + 1

    def forward(self, bnd, rho):
        mseqout = self.mseq(self.catter(bnd, rho))
        return self.eseq(mseqout.view(mseqout.size(0), -1))


def runOneEpochPA(model, dataset, iepoch, optimizer=None, scheduler=None,  iftrain=False, see=10):
    if iftrain:
        model.train()
    else:
        model.eval()

    lossHist = []

    ib = 0

    for batch in dataset:
        ib += 1
        target = torch.log(batch['PI_AB'] + 1e-6)
        bnd = batch['bnd']
        # bnd[0,0,:] -= 0.5
        # bnd[0,-1,:] -= 0.5
        # bnd[0, 1:-1,0] -= 0.5
        # bnd[0, 1:-1, -1] -= 0.5
        result = model(bnd, batch['rho']).view(target.size())

        if iftrain:
            optimizer.zero_grad()
        
        loss = F.mse_loss(result, target)
        lossHist.append(float(loss))
        if iftrain:
            loss.backward()
            optimizer.step()
            scheduler.step()

        if(ib % see == 0 and iftrain):
            print('Train Epoch %d :: Batch number %4d, loss [%.6e]' % (
                iepoch, ib, loss))

    return (lossHist, torch.mean(torch.tensor(lossHist)))


def runOneEpochPAStatic(model, dataset, iepoch, optimizer=None, scheduler=None,  iftrain=False, see=10):
    if iftrain:
        model.train()
    else:
        model.eval()

    lossHist = []

    ib = 0

    for batch in dataset:
        ib += 1
        # target = torch.log(batch['PI_AB'] + 1e-6)
        target = batch['PI_AB'] ** 0.5

        # bnd = batch['bnd']
        # # bnd[0,0,:] -= 0.5
        # # bnd[0,-1,:] -= 0.5
        # # bnd[0, 1:-1,0] -= 0.5
        # # bnd[0, 1:-1, -1] -= 0.5
        result = model(batch['rho']).view(target.size())

        if iftrain:
            optimizer.zero_grad()
        
        loss = F.mse_loss(result, target)
        lossHist.append(float(loss))
        if iftrain:
            loss.backward()
            optimizer.step()
            scheduler.step()

        if(ib % see == 0 and iftrain):
            print('Train Epoch %d :: Batch number %4d, loss \x1b[1;33m[%.6e]\x1b[0m' % (
                iepoch, ib, loss))

    return (lossHist, torch.mean(torch.tensor(lossHist)))
