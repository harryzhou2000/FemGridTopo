from typing import List, Tuple
import numpy
from numpy.lib.function_base import delete
from numpy.lib.scimath import sqrt
import scipy
import scipy.sparse
import scipy.signal
import pypardiso
import os
import copy



import pyoptsparse
import femRandInput


class planarElem:
    def __init__(self, nnode=4, nint=4) -> None:
        self.nnode = nnode
        self.nint = nint
        self.shape_funcs = []
        self.shape_funcs_dxi = []
        self.shape_funcs_deta = []

        if nnode == 4:
            self.shape_funcs.append(lambda xi, eta: +0.25 * (xi - 1.0) * (eta - 1.0))
            self.shape_funcs.append(lambda xi, eta: -0.25 * (xi + 1.0) * (eta - 1.0))
            self.shape_funcs.append(lambda xi, eta: +0.25 * (xi + 1.0) * (eta + 1.0))
            self.shape_funcs.append(lambda xi, eta: -0.25 * (xi - 1.0) * (eta + 1.0))

            self.shape_funcs_dxi.append(lambda xi, eta: +0.25 * (1.0) * (eta - 1.0))
            self.shape_funcs_dxi.append(lambda xi, eta: -0.25 * (1.0) * (eta - 1.0))
            self.shape_funcs_dxi.append(lambda xi, eta: +0.25 * (1.0) * (eta + 1.0))
            self.shape_funcs_dxi.append(lambda xi, eta: -0.25 * (1.0) * (eta + 1.0))

            self.shape_funcs_deta.append(lambda xi, eta: +0.25 * (xi - 1.0) * (1.0))
            self.shape_funcs_deta.append(lambda xi, eta: -0.25 * (xi + 1.0) * (1.0))
            self.shape_funcs_deta.append(lambda xi, eta: +0.25 * (xi + 1.0) * (1.0))
            self.shape_funcs_deta.append(lambda xi, eta: -0.25 * (xi - 1.0) * (1.0))
        else:
            raise Exception("invalid nnode")

        self.int_weights = []
        self.int_coords = []
        if nint == 4:
            a = 1.0 / numpy.sqrt(3)
            self.int_weights.append(1.0)
            self.int_coords.append([-a, -a])
            self.int_weights.append(1.0)
            self.int_coords.append([+a, -a])
            self.int_weights.append(1.0)
            self.int_coords.append([-a, +a])
            self.int_weights.append(1.0)
            self.int_coords.append([+a, +a])
        elif nint == 1:
            self.int_coords.append([0.0, 0.0])
            self.int_weights.append(2.0)
        else:
            raise Exception("invalid nint")

        self.NImr = numpy.zeros((nnode, nint))
        self.dNIdLimr = numpy.zeros((2, nnode, nint))

        for iint in range(nint):
            for inode in range(nnode):
                self.NImr[inode, iint] = self.shape_funcs[inode](
                    self.int_coords[iint][0], self.int_coords[iint][1]
                )
                self.dNIdLimr[0, inode, iint] = self.shape_funcs_dxi[inode](
                    self.int_coords[iint][0], self.int_coords[iint][1]
                )
                self.dNIdLimr[1, inode, iint] = self.shape_funcs_deta[inode](
                    self.int_coords[iint][0], self.int_coords[iint][1]
                )

        pass


class planarMat:
    def __init__(self, E=1.0, nu=0.3):
        self.E = E
        self.nu = nu
        self.D = numpy.eye(3, 3)
        self.D[2, 2] -= nu
        self.D[0, 1] = self.D[1, 0] = nu
        self.D *= E / (1 - nu ** 2)


class isoGridFem2D:
    def __init__(self, nx=100, ny=100, Lx=1.0, Ly=1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.xs = numpy.linspace(0, Lx, nx + 1)
        self.ys = numpy.linspace(0, Ly, ny + 1)
        self.xcs = (self.xs[:-1] + self.xs[1:]) * 0.5
        self.ycs = (self.ys[:-1] + self.ys[1:]) * 0.5
        self.xcm, self.ycm = numpy.meshgrid(self.xcs, self.ycs)
        self.xm, self.ym = numpy.meshgrid(self.xs, self.ys)

        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        self.npoint = (nx + 1) * (ny + 1)
        self.ndof = self.npoint * 2
        self.nelem = nx * ny

        self.VM = numpy.empty_like(self.xcm)
        self.uglob = numpy.empty((self.ndof))
        self.bglob = numpy.zeros(self.ndof)

        self.zeroT = 1e-3
        self.penalty = 3.0

        pass

    def SetElemMat(self, nnode=4, nint=4, E=1.0, nu=0.3) -> None:
        self.nnode = nnode
        self.nint = nint
        self.elem = planarElem(nnode=nnode, nint=nint)
        self.mat = planarMat(E=E, nu=nu)
        self.nelemdof = 2 * nnode
        pass

    def GetElemMat(self) -> Tuple[numpy.array, numpy.array, list]:
        Kmat = numpy.zeros((self.nnode * 2, self.nnode * 2))
        smatlist = []
        idxlist = []
        coords = numpy.zeros((self.nnode, 2))
        if self.nnode == 4:
            coords[0, :] = [0, 0]
            coords[1, :] = [self.dx, 0]
            coords[2, :] = [self.dx, self.dy]
            coords[3, :] = [0, self.dy]
        else:
            raise Exception("invalid nnode")

        for iint in range(self.nint):
            Jacobi = self.elem.dNIdLimr[:, :, iint] @ coords
            dJacobi = numpy.linalg.det(Jacobi)
            iJacobi = numpy.linalg.inv(Jacobi)
            gradU = iJacobi @ self.elem.dNIdLimr[:, :, iint]
            gradDisp = numpy.zeros((4, self.nnode * 2))
            # gradDisp @ elem_a = [dudx dudy dvdx dvdy]
            for inode in range(self.nnode):
                gradDisp[0:2, inode * 2 + 0] = gradU[0:2, inode]
                gradDisp[2:4, inode * 2 + 1] = gradU[0:2, inode]

            B = numpy.zeros((3, self.nnode * 2))
            B[0, :] = gradDisp[0, :]
            B[1, :] = gradDisp[3, :]
            B[2, :] = (gradDisp[1, :] + gradDisp[2, :]) * 0.5

            DB = self.mat.D @ B
            smatlist.append(DB)

            Kinc = numpy.transpose(B) @ DB

            Kmat += dJacobi * self.elem.int_weights[iint] * Kinc

        Kmat = 0.5 * (Kmat.transpose() + Kmat)

        smat = numpy.zeros_like(smatlist[0])
        for smati in smatlist:
            smat += smati
        smat /= len(smatlist)

        return (Kmat, smat, idxlist)

    def setBCs(self, fix: numpy.array, fx: numpy.array, fy: numpy.array):  # doing ref
        self.fix = fix
        self.fx = fx
        self.fy = fy

    def AssembleKbLocal(self) -> None:
        self.nkinsert = self.nelemdof ** 2 * self.nelem
        self.Ki = numpy.empty((self.nkinsert), dtype=numpy.int32)
        self.Kj = numpy.empty((self.nkinsert), dtype=numpy.int32)
        self.Kv = numpy.empty((self.nkinsert), dtype=numpy.float64)
        self.Ktemp, self.Bstemp, idxempty = self.GetElemMat()

        self.Klocal = []
        self.blocal = []
        self.idoflocal = []
        self.idxilocal = []
        self.idxjlocal = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                if self.nnode == 4:
                    ipx = numpy.array([ix, ix + 1, ix + 1, ix], dtype=numpy.int32)
                    ipy = numpy.array([iy, iy, iy + 1, iy + 1], dtype=numpy.int32)

                    ip = ipy * (self.nx + 1) + ipx
                    idofs = numpy.concatenate(
                        (ip.reshape((1, -1)) * 2, ip.reshape((1, -1)) * 2 + 1), axis=0
                    ).reshape((-1), order="F")
                    idxi = numpy.outer(idofs, numpy.ones((8), dtype=numpy.int32))
                    idxj = idxi.transpose()
                    idxi = idxi.reshape((-1))
                    idxj = idxj.reshape((-1))
                    # print(ipy)
                    Kc = copy.deepcopy(self.Ktemp)
                    Fixp = self.fix[ipy, ipx]
                    Bcx = self.fx[ipy, ipx]
                    Bcy = self.fy[ipy, ipx]
                    Bdof = numpy.concatenate(
                        (Bcx.reshape((1, -1)), Bcy.reshape((1, -1))), axis=0
                    ).reshape((-1), order="F")
                    Ffixdof = numpy.concatenate(
                        (Fixp.reshape((1, -1)), Fixp.reshape((1, -1))), axis=0
                    ).reshape((-1), order="F")
                    for idof in range(self.nelemdof):
                        if not numpy.isnan(Ffixdof[idof]):
                            ### uncomment for common fix bd
                            # Kline = Kc[:, idof].reshape((-1))
                            # for idof in range(self.nelemdof):
                            #     if(not numpy.isnan(Ffixdof[idof])):
                            #         Kline[idof] = 0.0
                            # Bdof -= Kline * Ffixdof[idof]
                            Bdof[idof] = 0.0
                            Kdiag = Kc[idof, idof]
                            Kc[idof, :] = 0.0
                            Kc[:, idof] = 0.0
                            Kc[idof, idof] = Kdiag
                    del idof
                    self.Klocal.append(copy.deepcopy(Kc))
                    self.blocal.append(copy.deepcopy(Bdof))
                    self.idxilocal.append(copy.deepcopy(idxi))
                    self.idxjlocal.append(copy.deepcopy(idxj))
                    self.idoflocal.append(copy.deepcopy(idofs))
                    # print(self.idoflocal[-1])

                else:
                    raise Exception("invalid nnode")

    def setRho(self, rho: numpy.array):  # doing ref
        self.rho = rho

    def AssembleRhoAb(self) -> None:
        nfill = 0
        self.bglob.fill(0.0)
        for iy in range(self.ny):
            for ix in range(self.nx):
                ie = iy * self.nx + ix
                # print(self.idoflocal[ie])
                self.bglob[self.idoflocal[ie]] += self.blocal[ie]
                self.Ki[nfill : nfill + self.nelemdof ** 2] = self.idxilocal[ie]
                self.Kj[nfill : nfill + self.nelemdof ** 2] = self.idxjlocal[ie]
                # self.Kv[nfill: nfill + self.nelemdof **
                #         2] = self.Klocal[ie].reshape((-1)) * self.rho[iy, ix]
                self.Kv[nfill : nfill + self.nelemdof ** 2] = self.Klocal[ie].reshape(
                    (-1)
                ) * numpy.power(self.rho[iy, ix], self.penalty)

                nfill += self.nelemdof ** 2
        self.Kglob = scipy.sparse.csr_matrix(
            (self.Kv, (self.Ki, self.Kj)), dtype=numpy.float64
        )

    def AssembleRhoATriplet(self) -> None:
        nfill = 0
        for iy in range(self.ny):
            for ix in range(self.nx):
                ie = iy * self.nx + ix
                self.Ki[nfill : nfill + self.nelemdof ** 2] = self.idxilocal[ie]
                self.Kj[nfill : nfill + self.nelemdof ** 2] = self.idxjlocal[ie]
                self.Kv[nfill : nfill + self.nelemdof ** 2] = self.Klocal[ie].reshape(
                    (-1)
                ) * numpy.power(self.rho[iy, ix], self.penalty)
                nfill += self.nelemdof ** 2

    def SolveU(self) -> None:
        self.uglob = pypardiso.spsolve(self.Kglob, self.bglob)
        self.u2 = self.uglob.reshape((2, -1), order="F")
        self.uarray = self.u2[0, :].reshape((self.ny + 1, self.nx + 1), order="C")
        self.varray = self.u2[1, :].reshape((self.ny + 1, self.nx + 1), order="C")

    def GetStress(self) -> None:
        self.VM.fill(0.0)
        for iy in range(self.ny):
            for ix in range(self.nx):
                ie = iy * self.nx + ix
                J2 = J2_2d(self.Bstemp @ self.uglob[self.idoflocal[ie]])
                self.VM[iy, ix] = J2


def J2_2d(sigma):
    return numpy.sqrt(
        (
            (sigma[0] - sigma[1]) ** 2
            + sigma[1] ** 2
            + sigma[0] ** 2
            + 6.0 * sigma[2] ** 2
        )
        / 2.0
    )



J2eps = 1e-5

def J2_2d_diff(sigma, J2):
    if(J2 < J2eps):
        return numpy.array([1.0,1.0,1.0])
    return (
        numpy.array(
            [sigma[0] - 0.5 * sigma[1], sigma[1] - 0.5 * sigma[0], 3.0 * sigma[2]]
        )
        / J2
    )


def Pnorm(x, p):
    if p > 100:
        return numpy.max(x)
    return numpy.power(numpy.sum(numpy.power(x, p)), 1.0 / p)


def Pnorm_diff(x, p):
    if p > 100:
        ret = numpy.zeros_like(x)
        idx = numpy.unravel_index(numpy.argmax(x), x.shape)
        ret[idx] = 1.0
        return ret
    return numpy.power(x, p - 1) * numpy.power(
        numpy.sum(numpy.power(x, p)), 1.0 / p - 1
    )


class isoGridFem2DOptFun(isoGridFem2D):
    def __init__(self, pVM=1000, nx=100, ny=100, Lx=1, Ly=1, nfilt = 5) -> None:
        super().__init__(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
        self.pVM = pVM
        self.dPI_VMdU = numpy.empty_like(self.uglob)
        self.dPI_VMdrho = numpy.empty_like(self.xcm)
        self.dPI_ABdrho = numpy.empty_like(self.xcm)

        # self.ABfilter = numpy.ones((3, 3))
        # self.ABfilter[:,1] = 1
        # self.ABfilter[1,:] = 1
        # self.ABfilter[1,1] = 1
        # self.ABfilter = self.ABfilter / numpy.sum(self.ABfilter)

        self.ABfilter = femRandInput.gaussianWin2(nfilt,nfilt)

        
        self.rhoSeq = []
        self.evalMax = -1
        self.neval = 0

    def SetElemMat(self, nnode=4, nint=4, E=1, nu=0.3) -> None:
        return super().SetElemMat(nnode=nnode, nint=nint, E=E, nu=nu)

    def setBCs(self, fix: numpy.array, fx: numpy.array, fy: numpy.array):
        return super().setBCs(fix, fx, fy)

    def AssembleKbLocal(self) -> None:
        return super().AssembleKbLocal()

    def EvalVM(self, rho, useVM = True, storeRhoSeq = False):
        if(self.neval > self.evalMax and self.evalMax > 0):
            print('Exceed Jump')
            return
        
        self.setRho(rho)
        self.AssembleRhoAb()
        self.SolveU()
        if(useVM):
            self.GetStress()
            self.PI_VM = Pnorm(self.VM, self.pVM)
            
        self.PI_AB = numpy.dot(self.uglob, self.bglob)

        self.neval += 1

        if storeRhoSeq:
            if useVM:
                self.rhoSeq.append((copy.deepcopy(rho),copy.deepcopy(self.uarray), copy.deepcopy(self.varray), copy.deepcopy(self.PI_AB), copy.deepcopy(self.VM)))
            else:
                self.rhoSeq.append((copy.deepcopy(rho),copy.deepcopy(self.uarray), copy.deepcopy(self.varray), copy.deepcopy(self.PI_AB)))



    def EvalVMdiff(self, rho, useVM = True, storeRhoSeq = False):  # refrence
        self.rho = rho
        if(self.neval > self.evalMax and self.evalMax > 0):
            self.dPI_ABdrho.fill(0.0)
            if(useVM):
                self.dPI_VMdrho.fill(0.0)
            return
        
        if(useVM):
            self.dPI_VMdVM = Pnorm_diff(self.VM, self.pVM)
            self.dPI_VMdU.fill(0.0)
            for iy in range(self.ny):
                for ix in range(self.nx):
                    ie = iy * self.nx + ix
                    sigma = self.Bstemp @ self.uglob[self.idoflocal[ie]]
                    dPI_VMdUlocal = (
                        J2_2d_diff(sigma, self.VM[iy, ix]) * self.dPI_VMdVM[iy, ix]
                    ) @ self.Bstemp  # * self.rho[iy,ix]
                    self.dPI_VMdU[self.idoflocal[ie]] += dPI_VMdUlocal
            self.dPI_VM_dKij_j = -pypardiso.spsolve(self.Kglob, self.dPI_VMdU)  # outer u
            self.dPI_VMdrho.fill(0.0)

            for iy in range(self.ny):
                for ix in range(self.nx):
                    ie = iy * self.nx + ix
                    # print(self.idoflocal[ie])
                    localdofs = self.idoflocal[ie]
                    ulocal = self.uglob[localdofs]
                    rholocalp = (
                        numpy.power(self.rho[iy, ix], self.penalty - 1) * self.penalty
                    )
                    dPI_VM_dKlocal = numpy.outer(ulocal, self.dPI_VM_dKij_j[localdofs])
                    dPI_VM_drholocal = (
                        numpy.sum(self.Klocal[ie] * dPI_VM_dKlocal) * rholocalp
                    )
                    self.dPI_VMdrho[iy, ix] = dPI_VM_drholocal
                    self.dPI_ABdrho[iy, ix] = ulocal.dot((self.Klocal[ie] @ ulocal)) * (
                        -rholocalp
                    )
        else: #only AB
            for iy in range(self.ny):
                for ix in range(self.nx):
                    ie = iy * self.nx + ix
                    # print(self.idoflocal[ie])
                    localdofs = self.idoflocal[ie]
                    ulocal = self.uglob[localdofs]
                    rholocalp = (
                        numpy.power(self.rho[iy, ix], self.penalty - 1) * self.penalty
                    )
                    self.dPI_ABdrho[iy, ix] = ulocal.dot((self.Klocal[ie] @ ulocal)) * (
                        -rholocalp
                    )

        if storeRhoSeq:
            if useVM:
                self.rhoSeq.append((copy.deepcopy(rho), copy.deepcopy(self.uarray), copy.deepcopy(self.varray), copy.deepcopy(self.VM)))
            else:
                self.rhoSeq.append((copy.deepcopy(rho), copy.deepcopy(self.uarray), copy.deepcopy(self.varray)))
        

        # dPI_AB_dKij_j = u, alas (u @ uT) @@ K === uT @ k @ u = dPI_AB_Drho

    def FilterABdiff(self):
        self.dPI_ABdrho = scipy.signal.correlate2d(
            self.dPI_ABdrho, self.ABfilter, mode="same", boundary="symm"
        )

    def Eval(self, rho, useVM = True):
        self.EvalVM(rho, useVM=useVM)
        self.EvalVMdiff(rho, useVM=useVM)


class topoStressOptimizer:
    def __init__(self) -> None:

        pass

    def InitFunc(
        self, fix, fx, fy, pVM=1000, nx=100, ny=100, Lx=1, Ly=1, E=1.0, nu=0.3,
    ) -> None:
        self.femF = isoGridFem2DOptFun(pVM, nx, ny, Lx, Ly)
        self.femF.SetElemMat(4, 4, E, nu)
        self.femF.setBCs(fix=fix, fx=fx, fy=fy)
        self.femF.AssembleKbLocal()
        self.ifRunning = False

    def InitRho(self, volume):
        self.vlim = volume
        self.rho = numpy.empty_like(self.femF.xcm)
        self.rho.fill(volume)
        self.ifRunning = False
        self.femF.Eval(self.rho)

    def Step(self, LR):
        self.gradValid = -(self.femF.dPI_VMdrho - numpy.mean(self.femF.dPI_VMdrho))
        if not self.ifRunning:
            self.LgradValid0 = numpy.sum(self.gradValid ** 2) ** (0.5)
            self.ifRunning = True
        self.MgradValid = numpy.max(numpy.abs(self.gradValid))
        self.gradValid *= LR / self.LgradValid0
        if self.MgradValid > 0.1:
            self.gradValid *= 0.1 / self.MgradValid
        self.rho += self.gradValid

        self.rho = numpy.maximum(self.rho, self.femF.zeroT)
        self.rho = numpy.minimum(self.rho, 1.000)
        self.rho = self.rho / numpy.mean(self.rho) * self.vlim
        self.femF.Eval(self.rho)

class topoStiffOptimizer:
    def __init__(self) -> None:

        pass

    def InitFunc(
        self, fix, fx, fy, pVM=1000, nx=64, ny=64, Lx=1, Ly=1, E=1.0, nu=0.3,
    ) -> None:
        self.femF = isoGridFem2DOptFun(pVM, nx, ny, Lx, Ly)
        self.femF.SetElemMat(4, 4, E, nu)
        self.femF.setBCs(fix=fix, fx=fx, fy=fy)
        self.femF.AssembleKbLocal()
        self.ifRunning = False

    def InitRho(self, volume):
        self.vlim = volume
        self.rho = numpy.empty_like(self.femF.xcm)
        self.rho.fill(volume)
        self.ifRunning = False
        self.femF.Eval(self.rho, useVM=False)

    

    def Step(self, LR):
        self.gradValid = -(self.femF.dPI_VMdrho - numpy.mean(self.femF.dPI_ABdrho ))
        if not self.ifRunning:
            # self.LgradValid0 = numpy.sum(self.gradValid ** 2) ** (0.5)
            self.MgradValid0 = numpy.abs(self.gradValid).max()
            self.LRscale = self.MgradValid0/LR * 0.05
            self.ifRunning = True
        self.MgradValid = numpy.max(numpy.abs(self.gradValid))
        # self.gradValid *= LR / self.LgradValid0
        self.gradValid *= LR * self.LRscale
        if self.MgradValid > 0.05:
            self.gradValid *= 0.05 / self.MgradValid
        self.rhonew =  self.rho + self.gradValid
        self.rhonew = numpy.maximum(self.rho, self.femF.zeroT)
        self.rhonew = numpy.minimum(self.rho, 1.000)

        self.rhoInc = self.rhonew - self.rho
        self.rhoInc = self.rhoInc - numpy.mean(self.rhoInc)
        # self.rho = self.rho / numpy.mean(self.rho) * self.vlim
        self.femF.Eval(self.rho, useVM=False)
        