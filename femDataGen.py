from typing import Dict, Pattern
import numpy as np
import femGTopo
import femRandInput
import pandas as pd
import os
import time
import re

def caseStamp():
    return '%d_%d'%(round(time.time()), os.getpid())

class femDataGenFilterCut:
    def __init__(self,nx=64,ny=64,passes=[32,40,48,56,64],nskew=9,maxskew=10,ndir=8) -> None:
        self.nx = nx
        self.ny = ny
        self.rhogen = femRandInput.randRhoFilterGen(nx=nx,ny=ny,passes=passes,nskew=nskew,
            maxskew=maxskew,ndirection=ndir)
        self.bndgen = femRandInput.randBoundaryFilter1DGen(nx=nx,ny=ny,passes=passes)

        self.data = [] #data[ibnd][0] = (fix,fx,fy), data[ibnd][1] = [rho0, rho1 ....], data[ibnd][2] = [[u0, v0], [u1, v1]]
        self.fixth = 0.0
        self.fixflt=0.9
        self.forcecent = 0.6
        self.forceflt=0.8
        self.rhocent=0.5
        self.rhoflt=0.5
        self.VMselMax =1e5
        self.dataSeq = []

        #for fix selection
        self.max_fixportion = 0.5
        self.min_fixportion = 0.2
        self.min_minufixinterval = 0.035
        self.min_maxufixinterval = 0.25
        pass

    def genNextBnd(self, fixth=0.0, fixflt=0.6, forcecent=0.6, forceflt=0.8)->bool:
        portion, minuinterval, maxuinterval = -1.0,-1.0,-1.0
        while((portion < self.min_fixportion or portion > self.max_fixportion)
            or (minuinterval < self.min_minufixinterval or maxuinterval < self.min_maxufixinterval)):
            self.bndgen.genNext()
            self.fixData,self.fixIdx = self.bndgen.getBinaryMat(th=fixth,fltrange=fixflt)
            portion, maxuinterval, minuinterval =  self.bndgen.examineBinaryPortion()
            print('Generated Fix Bnd F[%.2g] Iup[%.2g] Ilo[%.2g]'%(portion,maxuinterval,minuinterval))

        

        # self.fixIdx = self.fixData==1.0
        self.fix = np.empty_like(self.fixData)
        self.fix.fill(np.nan)
        self.fix[self.fixIdx] = 0.0
        

        self.bndgen.genNext()
        self.fx = self.bndgen.getRandCutDownMat(fltrange=forceflt,datalo=-forcecent,datahi=forcecent)
        self.bndgen.genNext()
        self.fy = self.bndgen.getRandCutDownMat(fltrange=forceflt,datalo=-forcecent,datahi=forcecent)

        self.fx[self.fixIdx] = 0.0
        self.fy[self.fixIdx] = 0.0

    def genNextRho(self, rhocent=0.5,rhoflt=0.5):
        self.rhogen.genNext()
        self.rho = self.rhogen.getRandCutDown(datalo=0.5-rhocent,datahi=0.5+rhocent,fltrange=rhoflt)

    def fillData(self, nbnd, nrho, old_sav=False):
        self.datanbnd = nbnd
        self.datanrho = nrho
        if old_sav:
            self.data = [] #data[ibnd][0] = (fix,fx,fy), data[ibnd][1] = [rho0, rho1 ....], data[ibnd][2] = [[u0, v0], [u1, v1]]
        else:
            self.bnddata = np.empty(shape=(self.ny+1,self.nx+1,3,1,nbnd), dtype=np.float32)
            self.rhodata = np.empty(shape=(self.ny,self.nx,1,nrho,nbnd), dtype=np.float32)
            self.resdata = np.empty(shape=(self.ny+1,self.nx+1,2,nrho,nbnd), dtype=np.float32)
            self.vmdata = np.empty(shape=(self.ny,self.nx,1,nrho,nbnd), dtype=np.float32)

        for ibnd in range(nbnd):
            if old_sav:
                bnddata = np.empty(shape=(self.ny+1,self.nx+1,3), dtype=np.float32)
                rhodata = np.empty(shape=(self.ny,self.nx,1,nrho), dtype=np.float32)
                resdata = np.empty(shape=(self.ny+1,self.nx+1,2,nrho), dtype=np.float32)
                vmdata = np.empty(shape=(self.ny,self.nx,1,nrho), dtype=np.float32)

            VMtest = 0.0
            itest = 0
            while(VMtest == 0.0):
                self.genNextBnd(fixth=self.fixth, fixflt=self.fixflt, 
                    forcecent=self.forcecent, forceflt=self.forceflt)
                fem = femGTopo.isoGridFem2D(nx = self.nx, ny = self.ny)
                fem.SetElemMat(4,4,1,0.3)
                fem.setBCs(self.fix,self.fx,self.fy)
                fem.AssembleKbLocal()
                

                self.genNextRho(rhocent=self.rhocent,rhoflt=self.rhoflt)
                fem.setRho(self.rho)
                fem.AssembleRhoAb()
                fem.SolveU()
                fem.GetStress()
                VMtest = fem.VM.max()
                itest+=1
                print('Ibnd %d test %d, maxvm  %g'%(ibnd, itest, VMtest))
            
            if old_sav:
                bnddata[:,:,0] = self.fixData
                bnddata[:,:,1] = self.fx
                bnddata[:,:,2] = self.fy
            else:
                self.bnddata[:,:,0,0,ibnd] = self.fixData
                self.bnddata[:,:,1,0,ibnd] = self.fx
                self.bnddata[:,:,2,0,ibnd] = self.fy

            for irho in range(nrho):
                
                VMtrial = 1e10
                ntrial = 0
                while (VMtrial > self.VMselMax):
                    self.genNextRho(rhocent=self.rhocent,rhoflt=self.rhoflt)
                    fem.setRho(self.rho)
                    fem.AssembleRhoAb()
                    fem.SolveU()
                    fem.GetStress()
                    VMtrial = fem.VM.max()
                    ntrial += 1
                
                if(old_sav):
                    rhodata[:,:,0,irho] = self.rho
                    resdata[:,:,0,irho] = fem.uarray
                    resdata[:,:,1,irho] = fem.varray
                    vmdata[:,:,0,irho] = fem.VM
                else:
                    self.rhodata[:,:,0,irho,ibnd] = self.rho
                    self.resdata[:,:,0,irho,ibnd] = fem.uarray
                    self.resdata[:,:,1,irho,ibnd] = fem.varray
                    self.vmdata[:,:,0,irho,ibnd] = fem.VM

                print("  Ibnd %4d ::: Irho %4d (ntri %2d) maxvm %.4e, maxu %.4e, maxv %.4e" % 
                (ibnd, irho, ntrial, fem.VM.max(), np.abs(fem.uarray).max(), np.abs(fem.varray.max())))
            if(old_sav):   
                self.data.append((bnddata,rhodata,resdata,vmdata))
        # if(not old_sav):
        #     self.dataSeq.append((self.bnddata,self.rhodata,self.resdata, self.vmdata))
        # pass

    def saveData(self, dir, old_sav=False, stamp = caseStamp()):
        os.makedirs(dir,exist_ok=True)
        if(old_sav):
            for ibnd in range(len(self.data)):
                    bndpath = os.path.join(dir,'bnd%010d.dat'%(ibnd))
                    rhopath = os.path.join(dir,'rho%010d.dat'%(ibnd))
                    respath = os.path.join(dir,'res%010d.dat'%(ibnd))
                    vmpath = os.path.join(dir,'vm%010d.dat'%(ibnd))
                    self.data[ibnd][0].tofile(bndpath)
                    self.data[ibnd][1].tofile(rhopath)
                    self.data[ibnd][2].tofile(respath)
                    self.data[ibnd][3].tofile(vmpath)
        else:
            brrvpath = os.path.join(dir,'brrv_%s.npz'%(stamp))
            np.savez(brrvpath, bnd = self.bnddata, 
                                        rho = self.rhodata, 
                                        res = self.resdata, 
                                        vm = self.vmdata)

    def loadData(self, dir, old_sav=False, seqTarget=0):
        dirs = os.listdir(dir)
        # pattern = r'[a-z]{2,}(\d*)\.dat'
        if old_sav:
            pattern = r'bnd(\d*)\.dat'
        
            fnameRe = re.compile(pattern=pattern)
            maxdigit = -1
            for fname in dirs:
                found = fnameRe.search(fname)
                if(found):
                    digit = int(found.group(1))
                    maxdigit = max([maxdigit, digit])

            nbnd = maxdigit + 1
            self.data = []
            for ibnd in range(nbnd):
                bndpath = os.path.join(dir,'bnd%010d.dat'%(ibnd))
                rhopath = os.path.join(dir,'rho%010d.dat'%(ibnd))
                respath = os.path.join(dir,'res%010d.dat'%(ibnd))
                vmpath = os.path.join(dir,'vm%010d.dat'%(ibnd))
                bnddata = np.reshape(np.fromfile(bndpath,dtype=np.float32,count=-1),newshape=(self.ny+1,self.nx+1,3))
                rhodata = np.reshape(np.fromfile(rhopath,dtype=np.float32,count=-1),newshape=(self.ny,self.nx,1,-1))
                resdata = np.reshape(np.fromfile(respath,dtype=np.float32,count=-1),newshape=(self.ny+1,self.nx+1,2,-1))
                vmdata = np.reshape(np.fromfile(vmpath,dtype=np.float32,count=-1),newshape=(self.ny,self.nx,1,-1))
                self.data.append((bnddata,rhodata,resdata,vmdata))
            print(maxdigit)

        else:
            self.loadDirs = []
            fnameRe = re.compile(pattern=r'.*\.npz')
            for fname in dirs:
                found = fnameRe.search(fname)
                if(found):
                    self.loadDirs.append(fname)
            if(seqTarget < 0):
                return
            if(seqTarget >= len(self.loadDirs)):
                raise(ValueError('seqTarget too large, max %d'%(len(self.loadDirs)-1)))
            self.dataLoadPath = os.path.join(dir,self.loadDirs[seqTarget])
            dataLoad = np.load(self.dataLoadPath)
            self.bnddata = dataLoad['bnd']
            self.rhodata = dataLoad['rho']
            self.resdata = dataLoad['res']
            self.vmdata = dataLoad['vm']
            self.datanbnd = np.size(self.bnddata, axis=4)
            self.datanrho = np.size(self.rhodata, axis=3)
            if(self.nx != np.size(self.rhodata, axis=1) or self.ny != np.size(self.rhodata, axis=0)):
                raise(Exception('data generator incompatible with datafile nx ny'))
    
    def old2new(self):
        nbnd = len(self.data)
        nrho = np.size(self.data[0][1], axis=3)

        self.bnddata = np.empty(shape=(self.ny+1,self.nx+1,3,1,nbnd), dtype=np.float32)
        self.rhodata = np.empty(shape=(self.ny,self.nx,1,nrho,nbnd), dtype=np.float32)
        self.resdata = np.empty(shape=(self.ny+1,self.nx+1,2,nrho,nbnd), dtype=np.float32)
        self.vmdata = np.empty(shape=(self.ny,self.nx,1,nrho,nbnd), dtype=np.float32)

        for ibnd in range(nbnd):
            self.bnddata[:,:,:,0,ibnd] = self.data[ibnd][0]
            self.rhodata[:,:,:,:,ibnd] = self.data[ibnd][1]
            self.resdata[:,:,:,:,ibnd] = self.data[ibnd][2]
            self.vmdata[:,:,:,:,ibnd] = self.data[ibnd][3]

    def getOneCase(self, ibnd, irho) -> Dict:
        if(ibnd <0 or ibnd >= self.datanbnd):
            raise(ValueError('ibnd out of range [0,%d)'%(self.datanbnd)))
        if(irho <0 or irho >= self.datanrho):
            raise(ValueError('irho out of range [0,%d)'%(self.datanrho)))
        
        caseret = {}

        caseret['fix'] = self.bnddata[:,:,0,0,ibnd]
        caseret['fx' ]= self.bnddata[:,:,1,0,ibnd]
        caseret['fy' ]= self.bnddata[:,:,2,0,ibnd]
        caseret['rho']= self.rhodata[:,:,0,irho,ibnd]
        caseret['u'  ]= self.resdata[:,:,0,irho,ibnd]
        caseret['v'  ]= self.resdata[:,:,1,irho,ibnd]
        caseret['vm' ]= self.vmdata[:,:,0,irho,ibnd]

        return caseret



            

        