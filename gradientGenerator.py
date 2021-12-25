from numpy import dtype
from torch.functional import meshgrid
from PP.models import networks
from PP.models.base_model import BaseModel
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt


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
    def __init__(self, firstGen=(4, 4), seq0=[16, 32, 16]):
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

        self.out = nn.Conv2d(lastend, 1, 1)

    def forward(self, bnd, rho):
        bnd2 = self.bndC2(bnd)
        rho1 = self.rhoC1(rho)
        dcat = torch.cat((bnd2, rho1), 1)
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
        result = model(batch['bnd'], batch['rho'])

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
    def __init__(self, firstGen=[2, 2], act=nn.ReLU()):
        super().__init__()
        self.bndC2 = nn.Conv2d(3, firstGen[0], 2)
        self.rhoC1 = nn.Conv2d(1, firstGen[1], 1)
        self.act = act

    def forward(self, bnd, rho):
        bnd2 = self.bndC2(bnd)
        rho1 = self.rhoC1(rho)
        dcat = torch.cat((bnd2, rho1), 1)
        return self.act(dcat)


class RhoBndCatCauchy(nn.Module):
    def __init__(self, firstGen=[2, 2], act=nn.ReLU()):
        super().__init__()
        self.bndC1 = nn.Conv2d(3, firstGen[0], 2)
        self.rhoC1 = nn.Conv2d(1, firstGen[1], 1)
        self.act = act

    def forward(self, bnd, rho):
        Hb = bnd.size(2)//2 * 2 + 1
        Wb = bnd.size(3)//2 * 2 + 1
        Hbnd = Hb//2
        Wbnd = Wb//2
        Is, Js = torch.meshgrid(torch.arange(-Hbnd, Hbnd+1, dtype=torch.float, device=bnd.device),
                                torch.arange(-Wbnd, Wbnd+1, dtype=torch.float, device=bnd.device), indexing='ij')
        Rsq = Is**2 + Js**2
        Rsq[Hbnd, Wbnd] = 1
        FiltCauchy = torch.sqrt((Hbnd**2+Wbnd**2)/Rsq).expand(1, 1, -1, -1)
        FiltCauchy /= FiltCauchy.sum()
        FiltZero = torch.zeros_like(FiltCauchy)
        FiltLayer0 = torch.cat((FiltCauchy, FiltZero, FiltZero), 1)
        FiltLayer1 = torch.cat((FiltCauchy, FiltZero, FiltZero), 1)
        FiltLayer2 = torch.cat((FiltCauchy, FiltZero, FiltZero), 1)
        Filt = torch.cat((FiltLayer0, FiltLayer1, FiltLayer2), 0)
        bndFilt = F.conv2d(bnd, Filt, padding='same')

        # plt.pcolor(bndFilt[0, 0, :, :].cpu().numpy())
        # plt.show()
        bnd2 = self.bndC1(bndFilt)
        rho1 = self.rhoC1(rho)
        dcat = torch.cat((bnd2, rho1), 1)
        return self.act(dcat)


class ConvDownSampler(nn.Module):
    def __init__(self, channelIn, channelOut, nds=2, act=nn.ReLU()):
        super().__init__()
        self.c = nn.Conv2d(channelIn, channelOut, nds, stride=nds, padding=0)
        self.act = act

    def forward(self, x):
        return self.act(self.c(x))


class ResNetForwardA(nn.Module):
    def __init__(self, channel, midchannel, ksize, act=nn.ReLU()):
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
    M = maxSize ** (4 * torch.arange(shape[1]//4, device=device)/shape[1])
    iH, iW = torch.meshgrid((torch.arange(shape[2], device=device), torch.arange(
        shape[3], device=device)), indexing='ij')
    Hc = torch.cos(
        iH.view((1, shape[2], shape[3])) / M.view((shape[1]//4, 1, 1)))
    Wc = torch.cos(
        iW.view((1, shape[2], shape[3])) / M.view((shape[1]//4, 1, 1)))
    Hs = torch.sin(
        iH.view((1, shape[2], shape[3])) / M.view((shape[1]//4, 1, 1)))
    Ws = torch.sin(
        iW.view((1, shape[2], shape[3])) / M.view((shape[1]//4, 1, 1)))
    E = torch.cat((Hc, Wc, Hs, Ws), dim=0)
    return E.expand(shape[0], -1, -1, -1)


class AddPosEncoding2D(nn.Module):
    def __init__(self, maxSize, ratioInit=1.0):
        super().__init__()
        self.maxSize = maxSize
        self.ratio = torch.nn.parameter.Parameter(torch.tensor(ratioInit))

    def forward(self, x):
        return x + PosEncoding2D(x.size(), self.maxSize, x.device) * self.ratio


class PixelwiseSelfAttention2d(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.att = nn.MultiheadAttention(dim, head, batch_first=True)
        self.bn = nn.BatchNorm2d(dim)
        self.dim = dim

    def forward(self, x, need_weights=False):
        # x : (N,C,H,W)
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        if(C != self.dim):
            raise ValueError(
                "PixelwiseSelfAttention2d: input channel %d needed" % (self.dim))
        xp = torch.permute(x.view((N, C, -1)), (0, 2, 1))
        ret = self.att(xp, xp, xp, need_weights=need_weights)
        # print(len(ret), need_weights)
        ret = (ret[0].permute(0, 2, 1).view((N, C, H, W)), ret[1])
        retx = ret[0] + x
        return self.bn(retx)


class PI_ABGenerator(nn.Module):
    def __init__(self, firstGen=[4, 4]):
        super().__init__()
        self.catter = RhoBndCatCauchy(firstGen)
        mlist = []
        elist = []
        act = nn.Softplus()
        mlist.append(ConvDownSampler(sum(firstGen), 16, 2, act))
        mlist.append(AddPosEncoding2D(64))
        mlist.append(PixelwiseSelfAttention2d(16, 1))
        mlist.append(ResNetForwardA(16, 8, 3, act))  # 32*32

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


class PI_ABGenertorStatic(nn.Module):
    def __init__(self):
        super().__init__()
        mlist = []
        elist = []
        act = nn.Softplus()
        mlist.append(ConvDownSampler(1, 4, 2, act))
        mlist.append(AddPosEncoding2D(64))
        # mlist.append(PixelwiseSelfAttention2d(4,1))
        mlist.append(ResNetForwardA(4, 2, 3, act))
        mlist.append(ResNetForwardA(4, 2, 3, act))
        mlist.append(ResNetForwardA(4, 2, 3, act))  # 32*32

        mlist.append(ConvDownSampler(4, 8, 2, act))
        # mlist.append(PixelwiseSelfAttention2d(8, 1))
        mlist.append(ResNetForwardA(8, 4, 3, act))
        mlist.append(ResNetForwardA(8, 4, 3, act))
        mlist.append(ResNetForwardA(8, 4, 3, act))  # 16*16

        mlist.append(ConvDownSampler(8, 16, 2, act))
        # mlist.append(PixelwiseSelfAttention2d(16, 1))
        mlist.append(ResNetForwardA(16, 8, 3, act))
        mlist.append(ResNetForwardA(16, 8, 3, act))
        mlist.append(ResNetForwardA(16, 8, 3, act))  # 8*8

        mlist.append(ConvDownSampler(16, 32, 2, act))
        # mlist.append(PixelwiseSelfAttention2d(32, 1))
        mlist.append(ResNetForwardA(32, 16, 3, act))
        mlist.append(ResNetForwardA(32, 16, 3, act))
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
            self.mseq.add_module('module_%d' % (i), m)
            i = i + 1
        i = 0
        self.eseq = nn.Sequential()
        for m in elist:
            self.eseq.add_module('module_%d' % (i), m)
            i = i + 1

    def forward(self, rho):
        mseqout = self.mseq(rho)
        return self.eseq(mseqout.view(mseqout.size(0), -1))


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
        target = torch.log(batch['PI_AB'] + 1e-6)
        # target = batch['PI_AB'] ** 0.5

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

####################################################


class InputConverter:
    def __init__(self, nchannel, nx, ny, mode='simple', device='cuda') -> None:
        self.device = device
        if mode == 'simple':
            self.ksize = (2, 2)
            self.kernel = torch.zeros(
                (nchannel, nchannel, 2, 2), device=device)
            for i in range(nchannel):
                self.kernel[i, i, :, :] = 1.0
            self.npadding = 0

        if mode == 'RBF':
            Lfilter = min(nx, ny)
            Rfilter = Lfilter//2
            iss, jss = torch.meshgrid([
                torch.arange(Rfilter * 2, dtype=torch.float,
                             device=self.device) - Rfilter + 0.5,
                torch.arange(Rfilter * 2, dtype=torch.float,
                             device=self.device) - Rfilter + 0.5
            ], indexing='ij')
            rss = torch.sqrt(iss ** 2 + jss ** 2) / Rfilter
            rss[rss > 1] = 1.0
            self.ksize = (Rfilter*2, Rfilter*2)
            self.filter = (1-rss)**2
            self.filter /= self.filter.sum()
            self.npadding = Rfilter-1
            self.kernel = torch.zeros(
                (nchannel, nchannel, Rfilter*2, Rfilter*2), device=device)
            for i in range(nchannel):
                self.kernel[i, i, :, :] = self.filter

        self.fconv = nn.Conv2d(nchannel, nchannel, self.ksize, padding=self.npadding,
                               padding_mode='zeros', device=device, bias=False)
        self.fconv.weight = nn.parameter.Parameter(
            self.kernel, requires_grad=False)
        self.fconv.requires_grad_(False)

    def __call__(self, x):
        return self.fconv(x)


def CompressMap(x, p = 0.25, s = 10):
    return (x.abs() ** p) * torch.tanh(x/s)

class RhoRes2Input:
    def __init__(self, nx, ny, bndmode='simple', resmode='strain', device='cuda') -> None:
        self.BndConverter = InputConverter(3, nx, ny, bndmode, device)
        self.resmode = resmode
        if(resmode == 'strain'):
            self.outChannel = 6
        if(resmode == 'displacement'):
            self.outChannel = 5

    def __call__(self, bnd, res):
        bndConverted = self.BndConverter(bnd)
        if self.resmode =='strain':
            dresdx = res[:, :, :, 1:]-res[:, :, :, 0:-1]
            dresdy = res[:, :, 1:, :]-res[:, :, 0:-1, :]
            dresdx = dresdx[:, :, 0:-1, :]+dresdx[:, :, 1:, :]
            dresdy = dresdy[:, :, :, 0:-1]+dresdy[:, :, :, 1:]
            exx = dresdx[:,0:1,:,:]
            eyy = dresdy[:,1:2,:,:]
            exy = ((dresdx[:, 1:2, :, :] + dresdy[:, 0:1, :, :]) * 0.5)
            eps = CompressMap(torch.cat((exx,eyy,exy),1))
            batchEpsMax = eps.abs().reshape((eps.size(0),-1)).max(1)[0].view((eps.size(0),1,1,1))
            eps /= batchEpsMax
            return torch.cat((bndConverted, eps), 1)
        if self.resmode == 'displacement':
            resc = (res[:, :, 1:, 1:  ] + res[:, :, 0:-1, 1:  ] +
                    res[:, :, 1:, 0:-1] + res[:, :, 0:-1, 0:-1])*0.25
            rescSize = resc.size()
            resc /= resc.reshape((rescSize[0],-1)).abs().max(1)[0].view((rescSize[0],1,1,1))
            return torch.cat((bndConverted,resc), 1)


class ResultGeneratorPP(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256',
                            dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float,
                                default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt,  channel_in=3,):
        super().__init__(opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            channel_in, 1, 64, 'unet_64', 'batch',
            False, 'normal', 0.02, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(channel_in + 1, 64, 'basic',
                                          3, 'batch', 'normal', 0.02, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss('vanilla').to(self.device)
            self.criterionL1 = torch.nn.SmoothL1Loss(beta=0.5)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(
            self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
