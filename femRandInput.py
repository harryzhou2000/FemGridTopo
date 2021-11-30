import numpy as np
import time

from numpy.core.function_base import linspace
import scipy.signal
from scipy.signal.filter_design import ellipord



rng = np.random.default_rng()

def set_seed(seed=0):
    global rng
    rng = np.random.default_rng(seed=seed)


def normBnd(bnd, cut = 2):
    return np.maximum(np.minimum( rng.normal(loc = 0.0, scale= bnd/cut), bnd),-bnd)

gWcut = 2
def gaussianWin2(nx,ny):
    xs = np.linspace(-gWcut,gWcut, nx)
    ys = np.linspace(-gWcut,gWcut,ny)
    ym,xm = np.meshgrid(ys,xs,indexing='ij')
    ret = np.exp(-(xm**2+ym**2))
    ret /= np.sum(ret)
    return ret

def gaussianWin2Oblique(nx,ny,Lx=1.0,Ly=1.0,oblique=0.0):
    rotM = np.array([[np.cos(oblique),-np.sin(oblique)],
    [np.sin(oblique),np.cos(oblique)]])
    ellip = np.diag([Lx,Ly])
    ellip = rotM @ ellip @ rotM.transpose()
    Lxr = ellip[0,0]
    Lyr = ellip[1,1]
    Lxy = 0.5*(ellip[0,1] + ellip[1,0])
    xs = np.linspace(-gWcut,gWcut, nx)
    ys = np.linspace(-gWcut,gWcut, ny)
    ym,xm = np.meshgrid(ys,xs,indexing='ij')
    ret = np.exp(-(xm**2*Lxr+ym**2*Lyr+2*xm*ym*Lxy))
    ret /= np.sum(ret)
    return ret

def gaussianWin1(nx):
    xs = np.linspace(-gWcut,gWcut,nx)
    ret = np.exp(-xs**2)
    ret /= ret.sum()
    return ret


def randBulk(nx,ny,Fxmax=1.0,win=gaussianWin2(10,10)):
    fx = rng.vonmises(mu=0,kappa=0.5,size=(nx,ny))
    fx = scipy.signal.correlate2d(
            fx, win, mode="same", boundary="wrap"
        )
    fx *= Fxmax/fx.max()
    return fx


def randBulkUni(nx,ny,Fxmax=1.0,win=gaussianWin2(10,10)):
    fx = rng.uniform(low=-Fxmax,high=Fxmax,
        size=(ny+np.size(win,0)-1,nx+np.size(win,1)-1))
    fx = scipy.signal.correlate2d(
            fx, win, mode="valid", boundary="wrap"
        )
    fx *= Fxmax/fx.max()
    return fx

def randLineUni(nx,Fxmax=1.0,win=gaussianWin1(10)):
    fxc = rng.uniform(low=-Fxmax,high=Fxmax,size=nx)
    # fx = np.concatenate((fxc,fxc[0:np.size(win,0)-1]),axis=0)
    fx = np.repeat(fxc,1 + np.ceil(np.float64(np.size(win,0))/nx))
    fx = scipy.signal.correlate(
        fx, win, mode="valid"
    )
    fx = fx[:nx]
    fx *= Fxmax/fx.max()
    return fx

def randDFSBulk(nx,ny,nfill):
    ret = np.zeros(shape=(ny,nx),dtype=np.int8)
    ind = np.array([0,0],dtype=np.int32)
    search = np.arange(4,dtype=np.int32)
    searchadd = np.array([[0,1],[1,0],[0,-1],[-1,0]])
    goodfront = []
    for iter in range(nfill):
        ret[ind[0],ind[1]] = 1
        np.random.shuffle(search)
        for dir in search:
            indseek = np.mod(ind + searchadd[dir],[nx,ny])
            if(ret[indseek[0],indseek[1]] == 0):
                goodfront.append(indseek)
        while(ret[ind[0],ind[1]] == 1):
            ind = goodfront.pop()
    
    return ret

def randDFSMomentumBulk(nx,ny,nfill):
    ret = np.zeros(shape=(ny,nx),dtype=np.int8)
    ind = np.array([0,0],dtype=np.int32)
    search = np.arange(4,dtype=np.int32)
    searchadd = np.array([[0,1],[1,0],[0,-1],[-1,0]])
    goodfront = []
    for iter in range(nfill):
        ret[ind[0],ind[1]] = 1
        np.random.shuffle(search)
        for dir in search:
            indseek = np.mod(ind + searchadd[dir],[nx,ny])
            if(ret[indseek[0],indseek[1]] == 0):
                goodfront.append(indseek)
        while(ret[ind[0],ind[1]] == 1):
            ind = goodfront.pop()
    
    return ret

class randFilterGen:
    def __init__(self, nx=64, ny=64, passes=[16,32,48,64], nskew=9, maxskew=10.0,ndirection=8) -> None:
        #filters: fil[ipass][iskew][idir]
        self.nx = nx
        self.ny = ny
        self.passes = passes
        self.nskew = nskew
        self.maxskew = maxskew
        self.skews = np.linspace(1.0,maxskew,nskew)
        self.ndirecion = ndirection
        self.directions = np.linspace(0.0,np.pi*(1-1.0/ndirection),ndirection)
        self.filts=[]
        for passsiz in self.passes:
            passfilts = []
            for skew in self.skews:
                skewfilts = []
                for dir in self.directions:
                    dirfilts = gaussianWin2Oblique(passsiz,passsiz,1.0,skew,dir)
                    skewfilts.append(dirfilts)
                passfilts.append(skewfilts)
            self.filts.append(passfilts)

        pass

    def genNextFilter(self)->None:
        self.iPass = rng.integers(0,len(self.passes))
        self.iSkew = rng.integers(0,self.nskew)
        self.iDir = rng.integers(0,self.ndirecion)
        self.currentfilt = self.filts[self.iPass][self.iSkew][self.iDir]
        pass

class randRhoFilterGen(randFilterGen):
    def __init__(self, nx=64, ny=64, passes=[16, 32, 48, 64], nskew=9, maxskew=10, ndirection=8) -> None:
        super().__init__(nx=nx, ny=ny, passes=passes, nskew=nskew, maxskew=maxskew, ndirection=ndirection)
        pass

    def genNext(self)->None:
        self.genNextFilter()
        self.data = randBulkUni(self.nx,self.ny,Fxmax=1.0, win=self.currentfilt)
        pass

    def getBinary(self,th=0.0):
        return self.data > th
    
    def getCutDown(self,lo=0.0,hi=1.0, datalo=-0.5,datahi=1.5):
        data = (self.data + 1.0)*((datahi-datalo)/2.0) +datalo
        data = np.maximum(data,lo)
        data = np.minimum(data,hi)
        return data

    def getRandCutDown(self, lo=1e-3,hi=1.0,datalo=-0.0,datahi=1.0, fltrange=0.5):
        # dataloU = datalo + rng.uniform(-fltrange,fltrange)
        # datahiU = datahi + rng.uniform(-fltrange,fltrange)
        dataloU = datalo + normBnd(fltrange)
        datahiU = datahi + normBnd(fltrange)
        return self.getCutDown(lo=lo,hi=hi,datalo=dataloU,datahi=datahiU)


class randBoundaryFilter1DGen(randFilterGen):
    def __init__(self, nx=64, ny=64, passes=[16, 32, 48, 64], nskew=9, maxskew=10, ndirection=8) -> None:
        super().__init__(nx=nx, ny=ny, passes=passes, nskew=nskew, maxskew=maxskew, ndirection=ndirection)
        self.nb = nx*2+ny*2
        self.filts1d = []
        for passsiz in passes:
            passfilts = gaussianWin1(passsiz)
            self.filts.append(passfilts)
        pass

    def genNext(self)->None:
        self.iPass1d = rng.integers(0,len(self.passes))
        self.currentfilt1d = self.filts1d[self.iPass1d]
        self.data = randLineUni(self.nb,Fxmax=1.0, win=self.currentfilt1d)
        pass

    def getBinary(self,th=0.0):
        return self.data > th
    
    def getBinaryRand(self, th=0.0, fltrange=0.6):
        # thU = th + rng.uniform(-fltrange,fltrange)
        thU = th + normBnd(fltrange)
        return self.getBinary(th=thU)

    def getBinaryMat(self, th=0.0, fltrange=0.6):
        self.dataBinaryCurrent = self.getBinaryRand(th=th,fltrange=fltrange)
        self.currentBinaryDouble = self.Data2Mat(self.dataBinaryCurrent,initfill=False)
        self.currentBinaryBool = self.currentBinaryDouble == 1.0
        return (self.currentBinaryDouble,self.currentBinaryBool)

    def getBinarySupportRho(self, lo=0.001, hi =1.0, datalo=-0.1,datahi=1.1, fltrange=0.3):
        dataloU = datalo + normBnd(fltrange)
        datahiU = datahi + normBnd(fltrange)
        self.genNextFilter()
        binFilted = scipy.signal.correlate2d(self.currentBinaryDouble,self.currentfilt,
            mode='same', boundary='fill' )
        binFiltedLo = binFilted.min()
        binFiltedHi = binFilted.max()

        binFilted = (binFilted - binFiltedLo) * ((datahi-datalo)/(binFiltedHi-binFiltedLo)) + datalo
        binFilted = np.minimum(binFilted, hi)
        binFilted = np.maximum(binFilted, lo)
        self.currentBinarySupport=binFilted
        return binFilted


    def getCutDown(self,lo=0.0,hi=1.0, datalo=-0.5,datahi=1.5):
        data = (self.data + 1.0)*((datahi-datalo)/2.0) +datalo
        data = np.maximum(data,lo)
        data = np.minimum(data,hi)
        return data

    def getRandCutDown(self, lo=-1.0,hi=1.0,datalo=-1.0,datahi=1.0, fltrange=0.5):
        # dataloU = datalo + rng.uniform(-fltrange,fltrange)
        # datahiU = datahi + rng.uniform(-fltrange,fltrange)
        dataloU = datalo + normBnd(fltrange)
        datahiU = datahi + normBnd(fltrange)
        return self.getCutDown(lo=lo,hi=hi,datalo=dataloU,datahi=datahiU)


    def getRandCutDownMat(self, lo =-1.0,hi=1.0,datalo=-1.0,datahi=1.0, fltrange=0.7):
        self.dataCutCurrent = self.getRandCutDown(lo=lo,hi=hi,datalo=lo,datahi=hi,fltrange=fltrange)
        return self.Data2Mat(self.dataCutCurrent,initfill=0.0)

    def examineBinaryPortion(self):
        portion = np.sum(self.dataBinaryCurrent)/np.size(self.dataBinaryCurrent)
        dBc = np.array(self.dataBinaryCurrent,dtype=np.int32)
        dBr = np.roll(dBc,-1,axis=0)
        dBdiff = dBr-dBc
        idxs = np.arange(np.size(dBdiff))
        selectbnd = dBdiff!=0
        if(np.sum(selectbnd)<1):
            if(dBc[0] == 0):
                return  (0.0, 1.0, 1.0)
            else:
                return  (1.0, 0.0, 0.0)

        diffbndc = dBdiff[selectbnd]
        idxbndc = idxs[selectbnd]
        idxbndr = np.roll(idxbndc,-1,axis=0)
        sizIntervals = idxbndr - idxbndc
        sizIntervals[-1] += np.size(dBdiff)
        sizinterval0 = sizIntervals[diffbndc<0] 
        maxInterval = sizinterval0.max()/np.size(self.dataBinaryCurrent)
        minInterval = sizinterval0.min()/np.size(self.dataBinaryCurrent)

        #return portion of 1, max,mininterval of 0
        return(portion, maxInterval, minInterval)

        
        



    def Data2Mat(self,data,initfill=0.0):
        ret = np.empty((self.ny+1,self.nx+1))
        ret.fill(initfill)
        ret[0,0:self.nx] = data[0:self.nx]
        ret[0:self.ny,-1] = data[self.nx:self.nx+self.ny]
        ret[-1,self.nx+1:0:-1] = data[self.nx+self.ny:self.nx+self.ny+self.nx]
        ret[self.ny+1:0:-1,0] = data[self.nx+self.ny+self.nx:self.nx+self.ny+self.nx+self.ny]
        return ret
        
        
    

