import numpy as np
import logging
from mpi4py import MPI

class Element(object):

    def __init__(self, dim):
        # self.comm = comm
        self.dim = dim
        self.dim_w = 1 if dim == 2 else 3
        self.dim_s = 3 if dim == 2 else 6
    
    def interpFun1D(self, Nodes, evalPoi):
        """Interpolate functions in 1D."""
        nevPoi = len(evalPoi)
        nNodes = len(Nodes)

        hFun = np.zeros((nevPoi, nNodes))
        dhFun = np.zeros((nevPoi, nNodes))

        Pat = np.ones((nNodes, nNodes))
        np.fill_diagonal(Pat, 0)

        for ievPoi in range(nevPoi):
            Num1 = evalPoi[ievPoi] * np.ones((nNodes, nNodes))
            Num2 = np.ones((nNodes, 1))*Nodes

            Num = Num1 - Num2
            np.fill_diagonal(Num, 1)
            prodNum = np.prod(Num, axis=1)

            Den = - Num2 + Num2.T
            np.fill_diagonal(Den, 1)

            prodDen = np.prod(Den, axis=1)

            Num3 = np.zeros((nNodes))
            for icol in range(nNodes):
                Num4 = Num.copy()
                Num4[:, icol] = Pat[:, icol]
                Num3 = Num3 + np.prod(Num4, axis=1)

            hFun[ievPoi, :] = prodNum / prodDen
            dhFun[ievPoi, :] = Num3 / prodDen
        return (hFun, dhFun)