from numpy import mod
from torch._C import device
import gradientGenerator
import dataReader
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import optimizer
from torch.serialization import load, save
import torch.nn.functional
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import time


class DummyParser:
    def __init__(self):
        self.continue_train = False
        self.verbose = True
        self.isTrain = True
        self.checkpoints_dir = 'PPsave'
        self.name = 'test3'
        self.preprocess = 'none'
        self.gpu_ids = [0]
        self.lr_policy = 'linear'
        self.epoch_count = 1
        self.n_epochs = 200
        self.n_epochs_decay = 200
        self.lr_decay_iters = 50
        self.lambda_L1 = 100.0

        self.lr = .0002

opt = DummyParser()
saveDir = os.path.join(opt.checkpoints_dir , opt.name)
if os.name == 'nt':  # windows or not
    os.system('mkdir '+saveDir)
    os.system('copy *.py '+saveDir)
else:
    os.system('mkdir -p '+saveDir)
    os.system('cp *.py '+saveDir)

Adevice = 'cuda'
Aconverter = gradientGenerator.InputConverter(3, 64, 64, 'RBF', Adevice)
AAconverter = gradientGenerator.RhoRes2Input(64, 64, 'simple', 'strain', Adevice)


model = gradientGenerator.ResultGeneratorPP(opt, channel_in=AAconverter.outChannel)
model.setup(opt)

# dataA = torch.rand(32, 3, 64, 64)
# dataB = torch.rand(32, 1, 64, 64)

# model.set_input({'A': dataA, 'B': dataB})
# model.optimize_parameters()
# model.update_learning_rate()
# print(model.get_current_losses())

# model.save_networks(0)



reader = dataReader.OptReader(32, preserve=['bnd', 'rho', 'res', 'res0'],
                              device=Adevice, trainPortion=1, delayed=True,randFlip=True)

reader.ReadFile('./data/test_optTrain', 0)
print(reader.size)

 #use training set
reader.state = 'train'


for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    tstart = time.perf_counter()
    i = 0
    errs = []
    for batch in iter(reader):
        # plt.pcolor(Aconverter(batch['bnd'])[0,0,:,:].cpu().numpy())
        # plt.show()
        # plt.pcolor(AAconverter(batch['bnd'], batch['res0'])[
        #            0, 3, :, :].cpu().numpy())
        # plt.colorbar()
        # plt.show()

        
        # model.set_input({'B':batch['rho'], 'A': Aconverter(batch['bnd'])})
        model.set_input({'B': batch['rho'], 'A': AAconverter(batch['bnd'], batch['res0'])})
        model.optimize_parameters()
        errs.append(model.get_current_losses())
        i += 1
        if( i % 10 == 0):
            
            print(
                'Epoch %4d Batch %4d, G_GAN \x1b[1;33m[%9.6g]\x1b[0m G_L1 \x1b[1;33m[%9.6g]\x1b[0m D_Real \x1b[1;33m[%9.6g]\x1b[0m D_Fake \x1b[1;33m[%9.6g]\x1b[0m' % (
                    epoch, i, errs[-1]['G_GAN'], errs[-1]['G_L1'], errs[-1]['D_real'], errs[-1]['D_fake']
                ))

    tend = time.perf_counter()
    print('Epoch %4d Time [%9.6g] ETA \x1b[1;36m[%9.6g]\x1b[0m' % (
        epoch, tend-tstart, (tend-tstart) * (opt.n_epochs + opt.n_epochs_decay -epoch)))

    model.update_learning_rate()

    if epoch % 10 == 0:
        model.save_networks(0)
        model.save_networks(epoch)
        print('\n=== ===\nSaved net %d\n=== ===\n' % epoch)


