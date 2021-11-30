import femGTopo
import numpy
import numpy.linalg

Nx = 100
Ny = 10
Lx = 10.
Ly = 2.

fem = femGTopo.isoGridFem2D(Nx, Ny, Lx, Ly)
fem.SetElemMat(4, 4, 1, 0.3)
# K,c,v =  fem.GetElemMat()
# v = numpy.linalg.eigvals(K)
# v.sort()


# a=  [1,2,3]
# a.extend(list(numpy.reshape(K,(-1))))
# b = a[0:4]
# b[0] = 3

fix = numpy.zeros((Ny+1, Nx+1))
fx = numpy.zeros_like(fix)
fy = numpy.zeros_like(fix)
fix *= numpy.nan
fix[:, 0] = 0
fx[:, -1] = 1.0/Ny
rho = numpy.ones((Ny, Nx))

fem.setBCs(fix, fx, fy)
fem.AssembleKbLocal()
fem.setRho(rho)
fem.AssembleRhoAb()
