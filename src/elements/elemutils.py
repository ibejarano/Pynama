"""Classes and help functions to be used within QUOTE.

   :Platform:
       Linux
   :License:
       To be decided!

.. moduleauthor:: Javier Quinteros <javier@gfz-potsdam.de>, GEOFON, GFZ Potsdam
.. moduleauthor:: Alejandro D. Otero <aotero@fi.uba.ar>, CONICET, Argentina
"""

from math import pi
from math import sqrt
import numpy as np
from numpy import linalg

import itertools
from functools import reduce
import operator
from elements.utilities import GaussPoint2D, GaussPoint3D

class Element(object):
    """Generic Element.

    :synopsis: Defines a generic Element with generic methods. Only the methods
           that do not change with the element typpe are implemented here.
           Classes implementing real Elements should inherit from this one.
           Attributes:
               nnode: Number of nodes per Element.
               gps: List of GaussPoints to be used in the integration.
               dim: Dimension of the problem.
               H: List with matrices with the shape functions evaluated at the
               corresponding Gauss point from "gps".
               dim: Dimension of the problem.
               elemType: String describing the element type.

    :platform: Any
    """

    def __init__(self):
        """Element constructor."""
        self.nnode = None
        self.gps = None
        self.dim = None
        self.H = None
        self.elemType = None

    def interpFunRS(self, r, s, dim=1):
        """Evaluate shape functions of a 4-nodes element @ r & s.

        :param r: Elemental coordinate R
        :type r: float
        :param s: Elemental coordinate S
        :type s: float
        :param dim: Dimension of the data to interpolate (1:scalar;
                    2:2D vector)
        :returns: An array with the shape functions
        :rtype: tuple
        """
        if dim == 1:
            # [h1 h2 h3 h4]
            shpF = np.array([(1+r)*(1+s)/4.0, (1-r)*(1+s)/4.0,
                             (1-r)*(1-s)/4.0, (1+r)*(1-s)/4.0])
        elif dim == 2:
            # [[h1 0  h2 0  h3 0  h4 0 ]
            #  [0  h1 0  h2 0  h3 0  h4]]
            shpF = np.array([[(1+r)*(1+s)/4.0, 0.0, (1-r)*(1+s)/4.0, 0.0,
                             (1-r)*(1-s)/4.0, 0.0, (1+r)*(1-s)/4.0, 0.0],
                            [0.0, (1+r)*(1+s)/4.0, 0.0, (1-r)*(1+s)/4.0,
                             0.0, (1-r)*(1-s)/4.0, 0.0, (1+r)*(1-s)/4.0]])
        else:
            raise Exception('Invalid dimension number %d' % dim)
        return shpF

    def interpFunDerRS(self, r, s, dim=1):
        """Evaluate shape functions derivatives of a 4-nodes element @ r & s.

        :param r: Elemental coordinate R
        :type r: float
        :param s: Elemental coordinate S
        :type s: float
        :param dim: Dimension of the data to interpolate (1:scalar;
                    2:2D vector)
        :returns: An array with the derivatives of the shape functions
        :rtype: tuple
        """
        if dim == 1:
            # [[h1,r h2,r h3,r h4,r]
            #  [h1,s h2,s h3,s h4,s]]
            dShpF = np.array([[(1+s)/4.0, -(1+s)/4.0, -(1-s)/4.0, (1-s)/4.0],
                              [(1+r)/4.0, (1-r)/4.0, -(1-r)/4.0, -(1+r)/4.0]])
        elif dim == 2:
            # [[h1,r 0    h2,r 0    h3,r 0    h4,r 0   ]
            #  [0    h1,s 0    h2,s 0    h3,s 0    h4,s]]
            dShpF = np.array(
                [[(1+s)/4.0, 0, -(1+s)/4.0, 0, -(1-s)/4.0, 0, (1-s)/4.0, 0],
                 [0, (1+r)/4.0, 0, (1-r)/4.0, 0, -(1-r)/4.0, 0, -(1+r)/4.0]])
        else:
            raise Exception('Invalid dimension number %d' % dim)
        return dShpF

    def setType(self, elemType):
        """Set the element type.

        :param elemType: Name of the Element type.
        :type elemType: string

        """
        self.elemType = elemType

    def _gaussPoints(self, N):
        """Calculate the properties of the Gauss points in 1D.

        :param N: Number of Gauss points in 1D.
        :type N: int
        :returns: Tuple with position of the points and their weight.
        :rtype: tuple

        """
        beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, N)) ** -2)
        T = np.diag(beta, 1) + np.diag(beta, -1)
        [W, V] = linalg.eig(T)
        i = np.argsort(W)
        x = W[i]
        w = 2 * V[0, i] ** 2
        # Symmetrize Gauss points
        x = (x - x[::-1]) / 2
        w = (w + w[::-1]) / 2
        return (x, w)

    def _lobattoPoints(self, N):
        """Calculate the properties of the Gauss-Lobatto points in 1D.

        :param N: Number of Gauss-Lobatto points in 1D.
        :type N: int
        :returns: Tuple with position of the points and their weight.
        :rtype: tuple

        """
        # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
        x = np.cos(np.linspace(0, pi, N))
        # The Legendre Vandermonde Matrix
        P = np.zeros((N, N))
        # Compute P_(N) using the recursion relation
        # Compute its first and second derivatives and
        # update x using the Newton-Raphson method.
        xold = 2
        while np.amax(np.abs(x - xold)) > (1e-15):
            xold = x
            P[:, 0] = 1
            P[:, 1] = x
            for k in range(2, N):
                P[:, k] = ((2 * k - 1) * x * P[:, k - 1] -
                           (k - 1) * P[:, k - 2]) / k
            x = xold - (x * P[:, N - 1] - P[:, N - 2]) / (N * P[:, N - 1])

        w = 2.0 / ((N - 1) * N * np.square(P[:, N - 1]))
        x = (x[::-1] - x) / 2
        w = (w[::-1] + w) / 2
        return (x, w)


class SpElem(Element):
    """Spectral element.

    :synopsis: Define a spectral element. Inherits from class:`~Element`.
           Attributes:
               All the attributes from class:`~Element` plus:
               NGL: Number of nodes per side.
               gps: List of GaussPoints to be used in the integration.
               dim: Dimension of the problem.
               H: List with matrices with the shape functions evaluated at the
               corresponding Gauss point from "gps".
               dim: Dimension of the problem.
               elemType: String describing the element type.
    :platform: Any
    """
    def __init__(self,NGL):
        self.NGL = NGL
        self.nnode = NGL ** 2
        self.nnodedge = NGL - 2
        self.nnodcell = (NGL - 2) ** 2

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

class SpElem2D(SpElem):
    """Spectral element in 2D.

    :synopsis: Define a spectral element in 2D. Inherits from class:`~Element`.
           Attributes:
               All the attributes from class:`~Element` plus:
               NGL: Number of nodes per side.
               gps: List of GaussPoints to be used in the integration.
               dim: Dimension of the problem.
               H: List with matrices with the shape functions evaluated at the
               corresponding Gauss point from "gps".
               dim: Dimension of the problem.
               elemType: String describing the element type.
    :platform: Any
    """
    def __init__(self, NGL):
        """Constructor of the SpElem2D class."""
        # NGL: Number of nodes in one direction. This is expanded in "nnode"
        # to the total number of nodes
        super().__init__(NGL)
        self.dim = 2
        self.elemType = 'Spectral2D(%d)' % NGL
        # First, create points in 1D
        nodes1D, operWei = self._lobattoPoints(NGL)
        gps1D, fullWei = self._gaussPoints(NGL) if NGL <= 3 else \
            self._lobattoPoints(NGL)
        gps_red1D, redWei = self._gaussPoints(NGL - 1)
        cnodes1D, cWei = self._lobattoPoints(2)
        self.indWCurl=[[0,0,1],[1,0,0]]
        self.indCurl=[[0,1,0],[0,0,1]]
        self.indBdiv=[[0,1],[1,2]]
        # Second, call the "interpFun2D" method to build the Gauss points,
        # the shape functions matrix and the matrix with its derivatives
        (self.H, self.Hrs, self.gps) = \
            self.interpFun2D(nodes1D, gps1D, fullWei)

        (self.HRed, self.HrsRed, self.gpsRed) = \
            self.interpFun2D(nodes1D, gps_red1D, redWei)

        (self.HOp, self.HrsOp, self.gpsOp) = \
            self.interpFun2D(nodes1D, nodes1D, operWei)

        (self.HCoo, self.HrsCoo, self.gpsCoo) = \
            self.interpFun2D(cnodes1D, gps1D, fullWei)

        (self.HCooRed, self.HrsCooRed, self.gpsCooRed) = \
            self.interpFun2D(cnodes1D, gps_red1D, redWei)

        (self.HCooOp, self.HrsCooOp, self.gpsCooOp) = \
            self.interpFun2D(cnodes1D, nodes1D, operWei)

        (self.HCoo1D, dh1D) = self.interpFun1D(cnodes1D, nodes1D)

    # FIXME: We need another interpFun2D for points in 2D with arbitrary (r,s)
    def interpFun2D(self, nodes1D, gps1D, gps1Dwei):
        """Interpolate functions in 2D."""
        (h1D, dh1D) = self.interpFun1D(nodes1D, gps1D)
        nNodes = len(nodes1D)
        ngps = len(gps1D)

        # Reorder polinomial roots
        # FIXME AAAAAAAAHHHHHHHHHH!!!!!! Indexing with 1 because it comes from
        # Matlab. This MUST be fixed so that ten lines below we are not forced
        # to substract "1"
        if nNodes > 1:
            Ind = np.zeros((nNodes, nNodes), dtype=int)
            Ind[np.ix_([0, -1], [0, -1])] = np.array([[2, 1], [3, 4]])

            if nNodes > 2:
                Ind[np.ix_([0], range(1, nNodes-1))] = \
                    np.array([range(5 + nNodes - 3, 4, -1)])
                Ind[np.ix_(range(1, nNodes - 1), [0])] = \
                    np.array([range(5 + nNodes - 2, 2 * nNodes + 1)]).T
                Ind[np.ix_([nNodes - 1], range(1, nNodes - 1))] = \
                    np.array([range(2 * nNodes + 1, 3 * nNodes - 1)])
                Ind[np.ix_(range(1, nNodes - 1), [nNodes - 1])] = \
                    np.array([range(4 * nNodes - 4, 3 * nNodes - 2, -1)]).T
                Ind[np.ix_(range(1, nNodes - 1), range(1, nNodes - 1))] = \
                    np.arange(4 * nNodes - 3, nNodes ** 2 + 1).reshape(
                    nNodes - 2, nNodes - 2).T
            Ind -= 1

            Permlst = Ind[::-1].T.reshape(1, np.prod(Ind.shape))[0].tolist()
        else:
            Permlst = [0]

        invPerm = [Permlst.index(val) for val in range(len(Permlst))]

        # Reorder evaluation points
        if ngps > 1:
            Ind2 = np.zeros((ngps, ngps), dtype=int)
            Ind2[np.ix_([0, -1], [0, -1])] = np.array([[2, 1], [3, 4]])

            if ngps > 2:
                Ind2[np.ix_([0], range(1, ngps-1))] = \
                    np.array([range(5 + ngps - 3, 4, -1)])
                Ind2[np.ix_(range(1, ngps - 1), [0])] = \
                    np.array([range(5 + ngps - 2, 2 * ngps + 1)]).T
                Ind2[np.ix_([ngps - 1], range(1, ngps - 1))] = \
                    np.array([range(2 * ngps + 1, 3 * ngps - 1)])
                Ind2[np.ix_(range(1, ngps - 1), [ngps - 1])] = \
                    np.array([range(4 * ngps - 4, 3 * ngps - 2, -1)]).T
                Ind2[np.ix_(range(1, ngps - 1), range(1, ngps - 1))] = \
                    np.arange(4 * ngps - 3, ngps ** 2 + 1).reshape(
                    ngps - 2, ngps - 2).T
            Ind2 -= 1

            Permlst2 = Ind2[::-1].T.reshape(1, np.prod(Ind2.shape))[0].tolist()
        else:
            Permlst2 = [0]

        invPerm2 = [Permlst2.index(val) for val in range(len(Permlst2))]

        # Interpolation functions H
        H = list()
        for doubleTern in itertools.product(h1D, h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            H.append(np.mat([auxRow[y] for y in invPerm]))

        # Derivatives of H wrt R & S
        Hrs = list()
        for doubleTern in itertools.product(dh1D, h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs.append(np.mat([[auxRow[y] for y in invPerm],
                               [0]*len(invPerm)]))

        for ind, doubleTern in enumerate(itertools.product(h1D, dh1D)):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs[ind][1, :] = [auxRow[y] for y in invPerm]

        # Gauss points
        gps = list()
        for c1 in range(len(gps1D)):
            for c2 in range(len(gps1D)):
                gps.append(GaussPoint2D(gps1D[c1], gps1D[c2],
                           gps1Dwei[c1] * gps1Dwei[c2]))

        H = [H[i] for i in invPerm2]
        Hrs = [Hrs[i] for i in invPerm2]
        gps = [gps[i] for i in invPerm2]

        return (H, Hrs, gps)


class SpElem3D(SpElem):
    """Spectral element in 3D.

    :synopsis: Define a spectral element in 3D. Inherits from class:`~Element`.
           Attributes:
               All the attributes from class:`~Element` plus:
               NGL: Number of nodes per side.
               gps: List of GaussPoints to be used in the integration.
               dim: Dimension of the problem.
               H: List with matrices with the shape functions evaluated at the
               corresponding Gauss point from "gps".
               dim: Dimension of the problem.
               elemType: String describing the element type.
    :platform: Any
    """
    def __init__(self, NGL):
        """Constructor of the SpElem3D class."""
        # NGL: Number of nodes in one direction. This is expanded in "nnode"
        # to the total number of nodes
        super().__init__(NGL)
        self.nnode = NGL ** 3
        self.nnodface = (NGL -2)**2
        self.nnodcell = (NGL - 2) ** 3
        self.dim=3
        self.elemType = 'Spectral3D(%d)' % NGL
        # Second 3D, call the interpFun2D method to build the Gauss points,
        # the shape functions matrix and the matrix with its derivatives
        nodes1D, operWei = self._lobattoPoints(NGL)
        gps1D, fullWei = self._gaussPoints(NGL) if NGL <= 3 else \
            self._lobattoPoints(NGL)
        gps_red1D, redWei = self._gaussPoints(NGL - 1)
        cnodes1D, cWei = self._lobattoPoints(2)
        self.indWCurl=[[0,2,1],[0,1,2],[1,2,0],[1,0,2],[2,1,0],[2,0,1]]
        self.indCurl=[[0,2,1],[0,1,2],[1,2,0],[1,0,2],[2,1,0],[2,0,1]]
        self.indBdiv=[[0,1,5],[1,2,3],[5,3,4]]
        (self.H, self.Hrs, self.gps) = \
            self.interpFun3D(nodes1D, gps1D, fullWei)

        (self.HRed, self.HrsRed, self.gpsRed) = \
            self.interpFun3D(nodes1D, gps_red1D, redWei)

        (self.HOp, self.HrsOp, self.gpsOp) = \
            self.interpFun3D(nodes1D, nodes1D, operWei)

        (self.HCoo, self.HrsCoo, self.gpsCoo) = \
            self.interpFun3D(cnodes1D, gps1D, fullWei)

        (self.HCooRed, self.HrsCooRed, self.gpsCooRed) = \
            self.interpFun3D(cnodes1D, gps_red1D, redWei)

        (self.HCooOp, self.HrsCooOp, self.gpsCooOp) = \
            self.interpFun3D(cnodes1D, nodes1D, operWei)

        (self.HCoo1D, dh1D) = self.interpFun1D(cnodes1D, nodes1D)

    # FIXME: We need another interpFun2D for points in 2D with arbitrary (r,s)
    def interpFun3D(self, nodes1D, gps1D, gps1Dwei):
        """Interpolate functions in 3D."""
        (h1D, dh1D) = self.interpFun1D(nodes1D, gps1D)
        nNodes = len(nodes1D)
        ngps = len(gps1D)

        invPerm = self.getSpOrder(nNodes)
        invPerm2 = self.getSpOrder(ngps)

        # Interpolation functions H
        H = list()
        for doubleTern in itertools.product(h1D, h1D ,h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            H.append(np.mat([auxRow[y] for y in invPerm]))

        # Derivatives of H wrt  R , S & T
        Hrs = list()
        for doubleTern in itertools.product(dh1D, h1D , h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs.append(np.mat([[auxRow[y] for y in invPerm],
                               [0]*len(invPerm),
                               [0]*len(invPerm)]))

        for ind, doubleTern in enumerate(itertools.product(h1D, dh1D , h1D)):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs[ind][1, :] = [auxRow[y] for y in invPerm]

        for ind, doubleTern in enumerate(itertools.product(h1D, h1D , dh1D)):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs[ind][2, :] = [auxRow[y] for y in invPerm]

        # Gauss points
        gps = list()
        for c1 in range(len(gps1D)):
            for c2 in range(len(gps1D)):
                for c3 in range(len(gps1D)):
                    gps.append(GaussPoint3D(gps1D[c1], gps1D[c2], gps1D[c3],
                           gps1Dwei[c1] * gps1Dwei[c2] * gps1Dwei[c3]))

        H = [H[i] for i in invPerm2]
        Hrs = [Hrs[i] for i in invPerm2]
        gps = [gps[i] for i in invPerm2]
        
        return (H, Hrs, gps)

    def getSpOrder(self, nPoints):
        # Reorder polinomial roots
        # FIXME AAAAAAAAHHHHHHHHHH!!!!!! Indexing with 1 because it comes from
        # Matlab. This MUST be fixed so that ten lines below we are not forced
        # to substract "1"
        if nPoints > 1:
            Ind3d = np.zeros((nPoints, nPoints,nPoints), dtype=int)
            Ind3d[np.ix_([0, -1],[0, -1],[0, -1])]  = np.array([[[8, 7], [5, 6]],[[2, 3], [1, 4]]])

            if nPoints > 2:
                nEdge = 8
                #edge1
                Ind3d[np.ix_([nPoints-1],range(1,nPoints-1),[0])] = np.mgrid[nEdge + nPoints-2 : nEdge :-1 ].reshape(1,nPoints-2,1)
                #edge2
                nEdge = nEdge + nPoints - 1
                Ind3d[np.ix_([nPoints-1],[0],range(1,nPoints-1))] = np.mgrid[nEdge : nEdge + nPoints - 2  ].reshape(1,1,nPoints-2)
                #edge3
                nEdge = nEdge + nPoints - 2
                Ind3d[np.ix_([nPoints-1],range(1,nPoints-1),[nPoints-1])] = np.mgrid[nEdge : nEdge + nPoints-2].reshape(1,nPoints-2,1)
                #edge4
                nEdge = nEdge + nPoints - 3
                Ind3d[np.ix_([nPoints-1],[nPoints-1],range(1,nPoints-1))] = np.mgrid[nEdge + nPoints-2 : nEdge : -1].reshape(1,1,nPoints-2)
                #edge5
                nEdge = nEdge + nPoints - 1
                Ind3d[np.ix_([0],[nPoints-1],range(1,nPoints-1))] = np.mgrid[ nEdge: nEdge + nPoints - 2 ].reshape(1,1,nPoints-2)
                #edge6
                nEdge = nEdge + nPoints  -3
                Ind3d[np.ix_([0],range(1,nPoints-1),[nPoints-1])] = np.mgrid[ nEdge + nPoints-2 : nEdge : - 1].reshape(1,nPoints-2,1)
                #edge7
                nEdge = nEdge + nPoints  - 2
                Ind3d[np.ix_([0],[0],range(1,nPoints-1))] = np.mgrid[ nEdge + nPoints - 2 : nEdge :-1].reshape(1,1,nPoints-2)

                #edge8
                nEdge = nEdge + nPoints  - 1
                Ind3d[np.ix_([0],range(1,nPoints-1),[0])] = np.mgrid[ nEdge : nEdge + nPoints-2].reshape(1,nPoints-2,1)
                #edge9
                nEdge = nEdge + nPoints - 3
                Ind3d[np.ix_(range(1,nPoints-1),[nPoints-1],[nPoints-1])] = np.mgrid[nEdge + nPoints - 2 : nEdge:-1 ].reshape(nPoints-2,1,1)
                #edge10
                nEdge = nEdge + nPoints - 1
                Ind3d[np.ix_(range(1,nPoints-1),[nPoints-1],[0])] = np.mgrid[ nEdge : nEdge + nPoints - 2].reshape(nPoints-2,1,1)
                #edge11
                nEdge = nEdge + nPoints - 3
                Ind3d[np.ix_(range(1,nPoints-1),[0],[0])] = np.mgrid[ nEdge + nPoints - 2 : nEdge : -1].reshape(nPoints-2,1,1)
                #edge12
                nEdge = nEdge + nPoints - 1
                Ind3d[np.ix_(range(1,nPoints-1),[0],[nPoints-1])] = np.mgrid[ nEdge : nEdge + nPoints - 2].reshape(nPoints-2,1,1)

                #face1
                nFace = (nPoints-2)**2
                nVert = nEdge + nPoints -2
                Ind3d[np.ix_([nPoints-1],range(1,nPoints-1),range(1,nPoints-1))] = np.mgrid[nVert : nVert+nFace].reshape(1,nPoints-2,nPoints-2)
                #face2
                nFace = (nPoints-2)**2
                nVert = nVert + nFace -1 
                Ind3d[np.ix_([0],range(1,nPoints-1),range(1,nPoints-1))] = np.mgrid[nVert+nFace : nVert : -1].reshape(1,nPoints-2,nPoints-2) #Invertir

                #face3
                nFace = (nPoints-2)**2
                nVert = nVert + nFace 
                Ind3d[np.ix_(range(1,nPoints-1),[nPoints-1],range(1,nPoints-1))] = np.mgrid[nVert+nFace : nVert : -1].reshape(nPoints-2,1,nPoints-2).T
                #face4
                nFace = (nPoints-2)**2
                nVert = nVert + nFace +1
                Ind3d[np.ix_(range(1,nPoints-1),[0],range(1,nPoints-1))] =  np.mgrid[nVert : nVert+nFace].reshape(nPoints-2,1,nPoints-2).T     #Invertir 
                
                #face5
                nFace = (nPoints-2)**2
                nVert = nVert +  nFace 
                Ind3d[np.ix_(range(1,nPoints-1),range(1,nPoints-1),[nPoints-1])] = np.mgrid[nVert : nVert+nFace].reshape(nPoints-2,nPoints-2,1)
                #face6
                nFace = (nPoints-2)**2
                nVert = nVert + nFace -1
                Ind3d[np.ix_(range(1,nPoints-1),range(1,nPoints-1),[0])] = np.mgrid[nVert+nFace : nVert : -1].reshape(nPoints-2,nPoints-2,1)
                #body
                nVert = nVert + nFace + 1 
                nBody = (nPoints-2)**3

                Ind3d[np.ix_(range(1,nPoints-1),range(1,nPoints-1),range(1,nPoints-1))] =\
                np.mgrid[nVert : nVert+nBody].reshape(nPoints-2,nPoints-2,nPoints-2)

            Ind3d -= 1
            Permlst = [Ind3d.T[perm].reshape(1,Ind3d.shape[0]**2)[0].tolist()[::-1] for perm in range(Ind3d.shape[0])]
            Permlst = np.array(Permlst).reshape(1, np.prod(Ind3d.shape))[0].tolist()
        else:
            Permlst = [0]

        invPerm = [Permlst.index(val) for val in range(len(Permlst))]
        return invPerm

class std4node2D(Element):
    """Standard 4 nodes 2D element.

    :synopsis: Defines a classical 4 nodes quadrilateral element in 2D.
        Inherits from class:`~Element`.
        Attributes:
               All the attributes from class:`~Element` plus:
               NGL: Number of nodes per side.
               gps: List of GaussPoints to be used in the integration.
               dim: Dimension of the problem.
               H: List with matrices with the shape functions evaluated at the
               corresponding Gauss point from "gps".
               dim: Dimension of the problem.
               elemType: String describing the element type.
    :platform: Any
    """

    def __init__(self):
        """Constructor of the std4node2D."""
        self.NGL = 2
        self.nnode = 4
        self.nnodedge = 0
        self.nnodcell = 0
        self.dim = 2
        self.elemType = 'std4node2D'

        thirdsqrt = sqrt(1/3.0)
        self.gps = [GaussPoint(thirdsqrt, thirdsqrt, 1.0),
                    GaussPoint(-thirdsqrt, thirdsqrt, 1.0),
                    GaussPoint(-thirdsqrt, -thirdsqrt, 1.0),
                    GaussPoint(thirdsqrt, -thirdsqrt, 1.0)]

    def evalGaussPointHrs(self, gp):
        """Build a matrix w/derivatives of the shape functions at Gauss points.

        :param gp: Gauss point where the derivatives of the shape functions
                   must be evaluated.
        :type gp: class:`~GaussPoint`
        :returns: Matrix with the evaluation of the derivatives of the shape
                  functions. The number of rows will be equal to the dimension
                  of the problem, while the number of columns will be equal to
                  the number of nodes in an element.
        :rtype: class:`~Numpy.Mat`
        """
        # Shape functions
        # h_1r =  (1 + s) / 4
        # h_2r = -(1 + s) / 4
        # h_3r = -(1 - s) / 4
        # h_4r =  (1 - s) / 4
        # h_1s =  (1 + r) / 4
        # h_2s =  (1 - r) / 4
        # h_3s = -(1 - r) / 4
        # h_4s = -(1 + r) / 4
        Hrs = [[(1 + gp.s) / 4, -(1 + gp.s) / 4,
                -(1 - gp.s) / 4,  (1 - gp.s) / 4],
               [(1 + gp.r) / 4,  (1 - gp.r) / 4,
                -(1 - gp.r) / 4, -(1 + gp.r) / 4]]
        return np.mat(Hrs)

    def evalGaussPointH(self, gp):
        """Build row vector w/the shape functions calculated at a Gauss point.

        :param gp: Gauss point where the shape functions must be evaluated.
        :type gp: class:`~GaussPoint`
        :returns: Vector with the evaluation of the shape functions. The length
                  of the vector will be equal to the number of nodes in an
                  element.
        :rtype: class:`~Numpy.Mat`

        """
        # Shape functions
        # h_1 = (1 + r) * (1 + s) / 4
        # h_2 = (1 - r) * (1 + s) / 4
        # h_3 = (1 - r) * (1 - s) / 4
        # h_4 = (1 + r) * (1 - s) / 4
        h_1 = (1 + gp.r) * (1 + gp.s) / 4.0
        h_2 = (1 - gp.r) * (1 + gp.s) / 4.0
        h_3 = (1 - gp.r) * (1 - gp.s) / 4.0
        h_4 = (1 + gp.r) * (1 - gp.s) / 4.0
        return np.mat([h_1, h_2, h_3, h_4])
