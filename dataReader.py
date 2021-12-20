import torch
import numpy as np
import re
import os
import random


class Reader:
    def __init__(self, nbatch, trainPortion=0.8, device='cuda', shuffle=True, preserve=['bnd', 'rho', 'da'],
        delayed = False) -> None:
        self.nbatch = nbatch
        self.device = device
        self.shuffle = shuffle
        self.preserve = preserve
        self.trainPortion = trainPortion
        self.delayed = delayed
        if(self.delayed):
            self.deviceLoad = 'cpu'
        else:
            self.deviceLoad = self.device

        pass

    def ReadFile(self, dir, seqTarget=0) -> bool:
        dirs = os.listdir(dir)
        self.loadDirs = []
        fnameRe = re.compile(pattern=r'.*\.npz')
        for fname in dirs:
            found = fnameRe.search(fname)
            if(found):
                self.loadDirs.append(fname)
        if seqTarget >= len(self.loadDirs):
            return False
        self.dataLoadPath = os.path.join(dir, self.loadDirs[seqTarget])
        dataLoad = np.load(self.dataLoadPath)
        self.bnddata = dataLoad['bnd']
        self.rhodata = dataLoad['rho']
        # self.resdata = dataLoad['res']
        # self.vmdata = dataLoad['vm']
        # self.drhodABdata = dataLoad['da']

        self.loadedData = {}
        for key in self.preserve:
            if (key == 'PI_AB'): 
                hasres = False
                hasbnd = False
                for key in self.preserve:
                    if key == 'res':
                        hasres = True
                    if key == 'bnd':
                        hasbnd = True
                    if key == 'PI_AB':
                        break
                if(not(hasres and hasbnd)):
                    raise ValueError("self.preserve not meeting PI_AB requirements")
                continue
            self.loadedData[key] = torch.tensor(
                dataLoad[key], device=self.deviceLoad, requires_grad=False).permute(4, 3, 2, 0, 1) 
                # new index order: [ibnd,irho,ichannel,H,W]
            

        self.datanbnd = np.size(self.bnddata, axis=4)
        self.datanrho = np.size(self.rhodata, axis=3)
        self.size = self.datanbnd * self.datanrho

        self.trainsize = int(np.ceil(self.size * self.trainPortion))
        self.testsize = self.size - self.trainsize
        self.splitShuffle = torch.randperm(self.size)

        self.state = 'train'
        self.currentBatch = {}

    def __iter__(self):
        self.iiter = 0
        if(self.state == 'train'):
            self.startiter = 0
            self.niter = self.trainsize
        elif(self.state == 'test'):
            self.startiter = self.trainsize
            self.niter = self.testsize
        else:
            raise ValueError('Bad state')
        if(self.shuffle and self.state == 'train'):
            self.iterseq = torch.randperm(
                self.niter, dtype=torch.long, device=self.deviceLoad)+self.startiter
        else:
            self.iterseq = torch.arange(
                self.niter, dtype=torch.long, device=self.deviceLoad)+self.startiter
        return self

    def __next__(self):
        if(self.iiter < self.niter):
            for key in list(self.currentBatch.keys()):
                del(self.currentBatch[key])
            take = self.splitShuffle[
                self.iterseq[self.iiter: self.iiter + self.nbatch]]
            if(self.device == 'cpu'):
                ibnds = take // self.datanrho
            else:
                ibnds = torch.div(take, self.datanrho, rounding_mode='trunc')
            irhos = take % self.datanrho
            dataBatch = {}
            dataBatch['ibnds'] = ibnds
            dataBatch['irhos'] = irhos
            for key in self.preserve:
                if(key == 'PI_AB'):
                    dataBatch['PI_AB'] = (dataBatch['bnd'][:, 1:3, :, :] * dataBatch['res']).sum((1,2,3)).to(self.device)
                    continue
                if(key != 'bnd'):
                    dataBatch[key] = self.loadedData[key][ibnds, irhos, :, :, :].to(self.device)
                else:
                    dataBatch[key] = self.loadedData[key][ibnds,
                                                          torch.zeros_like(ibnds, dtype=torch.long), :, :, :].to(self.device)

            self.iiter += self.nbatch
            self.currentBatch = dataBatch
            return dataBatch
        else:
            raise StopIteration


if __name__ == "__main__":
    print("testDataReader")
    reader = Reader(32, device='cuda')
    reader.ReadFile('./data/test_optTrain', 0)
    

    print(reader.size)
    i = 0
    reader.state = 'test'
    for batch in iter(reader):
        print(i, batch.keys())
        print(batch['rho'][0, 0, 0, 0])
        i += 1
    i = 0
    for batch in iter(reader):
        print(i, batch.keys())
        print(batch['rho'][0, 0, 0, 0])
        i += 1

        
    ##33333333333333333
    ##################

    #### note device and nbatch is set here, 
    # trainPortion=0.8 means 80% of data is used as training while rest are testing 
    reader = Reader(32, preserve=['bnd','rho','res','PI_AB'], 
        device='cpu', trainPortion=0.8, delayed=True)

    #### param1: path to data folder, param2: number of wanted datafile
    reader.ReadFile('./data/test_optTrain', 0)

    #use training set
    reader.state = 'train'
    i = 0
    for batch in iter(reader):
        print(i, batch.keys())
        print(batch['rho'].size(), batch['bnd'].size(), batch['PI_AB'].size())
        i+=1

    #use testing set
    reader.state = 'test'
    i = 0
    for batch in iter(reader):
        print(i, batch.keys())
        print(batch['rho'].size(), batch['bnd'].size(), batch['PI_AB'].size())
        i+=1

    
