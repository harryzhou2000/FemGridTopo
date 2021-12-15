import torch
import numpy as np
import re
import os
import random


class Reader:
    def __init__(self, nbatch, trainPortion=0.8, device='cuda', shuffle=True, preserve=['bnd', 'rho', 'da']) -> None:
        self.nbatch = nbatch
        self.device = device
        self.shuffle = shuffle
        self.preserve = preserve
        self.trainPortion = trainPortion
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
            self.loadedData[key] = torch.tensor(
                dataLoad[key], device=self.device).permute(4, 3, 2, 1, 0)
            

        self.datanbnd = np.size(self.bnddata, axis=4)
        self.datanrho = np.size(self.rhodata, axis=3)
        self.size = self.datanbnd * self.datanrho

        self.trainsize = int(np.ceil(self.size * self.trainPortion))
        self.testsize = self.size - self.trainsize
        self.splitShuffle = torch.randperm(self.size)

        self.state = 'train'

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
                self.niter, dtype=torch.long, device=self.device)+self.startiter
        else:
            self.iterseq = torch.arange(
                self.niter, dtype=torch.long, device=self.device)+self.startiter
        return self

    def __next__(self):
        if(self.iiter < self.niter):
            take = self.splitShuffle[
                self.iterseq[self.iiter: self.iiter + self.nbatch]]
            ibnds = torch.div(take, self.datanrho, rounding_mode='trunc')
            irhos = take % self.datanrho
            dataBatch = {}
            dataBatch['ibnds'] = ibnds
            dataBatch['irhos'] = irhos
            for key in self.preserve:
                if(key != 'bnd'):
                    dataBatch[key] = self.loadedData[key][ibnds, irhos, :, :, :]
                else:
                    dataBatch[key] = self.loadedData[key][ibnds,
                                                          torch.zeros_like(ibnds, dtype=torch.long), :, :, :]

            self.iiter += self.nbatch
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
    for batch in iter(reader):
        print(i, batch.keys())
        print(batch['rho'][0, 0, 0, 0])
        i += 1
