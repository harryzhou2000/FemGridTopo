import torch
import torch.nn as nn
from torch.nn.utils.rnn import invert_permutation, pack_padded_sequence
from torch.optim import optimizer
from torch.serialization import save
import torch.nn.functional
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import dataReader
import gradientGenerator

r'''
 should call after the trainOne.py has successfully exited
'''

torch.manual_seed(124)
params = {}
params['batch_size'] = 32
params['device'] = 'cuda'
params['lr'] = 1.0e-1
params['nepoch'] = 30
params['gamma'] = 0.8
params['weight_decay'] = 0e-4
params['strideSave'] = 1
selectC = -1


loader1 = dataReader.Reader(params['batch_size'], preserve=['bnd', 'rho', 'res', 'PI_AB'],
                            device=params['device'], trainPortion=1/8, delayed=False)
loader1.ReadFile('data/test2', -1)


runs = os.listdir('savesPGA')

casename = runs[selectC]

print('=== === loading case ' + casename)

saveDir = os.path.join('savesPGA', casename)

model = torch.load(os.path.join(saveDir, 'Original.pt'))

history = torch.load(os.path.join(saveDir, 'History.pt'))

select = torch.argmin(torch.tensor(history['lossEV']))

print('Selected %d, validation loss %g' % (select, history['lossEV'][select]))

selectStateDict = torch.load(os.path.join(
    saveDir, history['saveNames'][select]))

model.load_state_dict(selectStateDict)


Llist, lepoch = gradientGenerator.runOneEpochPA(
    model, iter(loader1), 0)
print('\n=== Test DONE ===')
print('Test loss [%.6e]\n' %
      (lepoch))

fout = open(os.path.join(saveDir, '_TestResult.txt'), 'w')

fout.write('Test loss [%.6e]\n' %
           (lepoch))
fout.close()


print(history.keys())

sz = 6

plt.figure('loss')
plt.plot(np.array(history['lossE']), label='train')
plt.plot(history['lossEV'], label='validation')
plt.xlabel('Epoch')
plt.title('Loss')
lgd = plt.legend()
plt.grid(which='both')
f = plt.gcf()
f.set_size_inches(sz, sz*0.7)
plt.savefig(os.path.join(saveDir, 'Loss.jpg'))
