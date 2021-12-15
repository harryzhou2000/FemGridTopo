import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as visMod
import time, os

def caseStamp():
    return '%d_%d' % (round(time.time()), os.getpid())

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
                     self.convB2(self.act(self.convB1(x))) + self.poolConv(self.poolA(x)))
        return y


class gradientGenA(nn.Module):
    def __init__(self, firstGen=(4,4), seq0 = [16,16,16]):
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
        lossHist.append(loss)
        if iftrain:
            loss.backward()
            optimizer.step()
            scheduler.step()


        if(ib % see == 0 and iftrain):
            print('Train Epoch %d :: Batch number %4d, loss [%.6e]' % (
                iepoch, ib, loss))
        

    return (lossHist, torch.mean(torch.tensor(lossHist)))
