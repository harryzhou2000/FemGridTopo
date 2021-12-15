from numpy import mod
import gradientGenerator
import dataReader



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

torch.manual_seed(124)
params = {}
params['batch_size'] = 64
params['device'] = 'cuda'
params['lr'] = 1.0e-1
params['nepoch'] = 20
params['gamma'] = 0.8
params['weight_decay'] = 0e-4
params['strideSave'] = 1

#####################################
loader = dataReader.Reader(
    params['batch_size'], device=params['device'], trainPortion=1, preserve=['bnd', 'rho', 'da'])
loader.ReadFile('./data/test_optTrain', 0)


print('=== Data Loaded ===')


#####################################
# set model and train

model = gradientGenerator.gradientGenA().to(params['device'])

print('\n#params = %d\n' %(sum([param.nelement() for param in model.parameters()])))

# optimizer = torch.optim.Adadelta(model.unfrozenParameters(), lr=params['lr'], weight_decay=1e-3)
optimizer = torch.optim.Adagrad(model.parameters(
), lr=params['lr'], weight_decay=params['weight_decay'])
scheduler = lr_scheduler.CyclicLR(optimizer, params['lr']*0.9, params['lr'],
                                  step_size_up=32,  gamma=params['gamma'], cycle_momentum=False)

# for item in model.parameters():
#     print(item)

lossHist = []
lossHistEpoch = []
lossHistEpochVal = []


saveDir = os.path.join('saves', gradientGenerator.caseStamp())
os.system('mkdir '+saveDir)
if os.name =='nt': # windows or not
    os.system('copy *.py '+saveDir)
else:
    os.system('cp *.py '+saveDir)
torch.save(params, os.path.join(saveDir, 'Params.pt'))
torch.save(model, os.path.join(saveDir, 'Original.pt'))

saveNames = []
isave = torch.arange(0,params['nepoch'],params['strideSave'])

for iepoch in range(params['nepoch']):
    print('\n\n=== === === === === === === === === === ===')
    start = time.perf_counter()
    loader.state = 'train'
    Llist, lepoch = gradientGenerator.runOneEpoch(
        model, iter(loader), iepoch, optimizer=optimizer, scheduler=scheduler, iftrain=True, see=50)
    print('\n=== Epoch DONE ===')
    print('Train Epoch %d :: loss [%.6e]\n' % (
        iepoch, lepoch))
    lossHist.extend(Llist)
    lossHistEpoch.append(lepoch)
    if any(isave == iepoch):
        saveNames.append('DictSav%04d' % (iepoch))
        torch.save(model.state_dict(), os.path.join(saveDir, saveNames[-1]))
        print('::: Saved Model :::')
    # loader.state = 'test'
    # Llist, lepoch = gradientGenerator.runOneEpoch(
    #     model, iter(loader), iepoch, iftrain=False)
    # print('\n=== Validation DONE ===')
    # print('Validation :: :: loss [%.6e]\n' % (
    #     lepoch))
    print('=== === === Epoch Time [%12.6g] === === ==='%(time.perf_counter()-start))
    print('=== === === === === === === === === === ===\n\n')
    lossHistEpochVal.append(lepoch)



torch.save({'loss': lossHist, 
            'lossE': lossHistEpoch, 
            'lossEV': lossHistEpochVal,
            'saveNames': saveNames},
           os.path.join(saveDir,'History.pt'))

