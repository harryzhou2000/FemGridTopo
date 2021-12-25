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
        self.isTrain = False
        self.checkpoints_dir = 'PPsave'
        self.name = 'test3'
        self.preprocess = 'none'
        self.gpu_ids = [0]
        self.lr_policy = 'linear'
        self.epoch_count = 1
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.lr_decay_iters = 50
        self.lambda_L1 = 100.0
        self.load_iter = 0
        self.epoch = 0


opt = DummyParser()
saveDir = os.path.join(opt.checkpoints_dir, opt.name)
# if os.name == 'nt':  # windows or not
#     os.system('mkdir '+saveDir)
#     os.system('copy *.py '+saveDir)
# else:
#     os.system('mkdir -p '+saveDir)
#     os.system('cp *.py '+saveDir)
Adevice = 'cuda'
Aconverter = gradientGenerator.InputConverter(3, 64, 64, 'RBF', Adevice)
AAconverter = gradientGenerator.RhoRes2Input(
    64, 64, 'simple', 'strain', Adevice)
model = gradientGenerator.ResultGeneratorPP(opt, channel_in=AAconverter.outChannel)
model.setup(opt)

# dataA = torch.rand(32, 3, 64, 64)
# dataB = torch.rand(32, 1, 64, 64)

# model.set_input({'A': dataA, 'B': dataB})
# model.optimize_parameters()
# model.update_learning_rate()
# print(model.get_current_losses())

# model.save_networks(0)



reader = dataReader.OptReader(1, preserve=['bnd', 'rho', 'res', 'res0'],
                              device=Adevice, trainPortion=0, delayed=True)

reader.ReadFile('./data/test_optTest', 1)
print(reader.size)

#use training set
reader.state = 'test'




for epoch in range(1):

    i = 0
    errs = []
    for batch in iter(reader):
        # plt.pcolor(Aconverter(batch['bnd'])[0,0,:,:].cpu().numpy())
        # plt.show()

        # model.set_input({'B': batch['rho'], 'A': Aconverter(batch['bnd'])})
        model.set_input(
            {'B': batch['rho'], 'A': AAconverter(batch['bnd'], batch['res0'])})
        model.forward()

        out = model.fake_B
        
        i += 1
        if(i % 1 == 0):
            plt.pcolor(batch['rho'][0,0,:,:].cpu().detach().numpy())
            plt.colorbar()
            plt.show()
            plt.pcolor(out[0, 0, :, :].cpu().detach().numpy())
            plt.colorbar()
            plt.show()



