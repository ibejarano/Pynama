import numpy as np
from numpy.linalg import inv, det
import itertools
from functools import reduce
import operator
from .element import Element
from .utilities import generateGaussPoints2D, generateGaussPoints3D, lobattoPoints, gaussPoints

class Spectral(Element):
    """Spectral element.
    :synopsis: Define a spectral element. Inherits from class:`~Element`.
           Attributes:
               ngl: Number of nodes per side.
               gps: List of GaussPoints to be used in the integration.
               elemType: String describing the element type.
    """
    def __init__(self, ngl, dim):
        super().__init__(dim)
        self.ngl = ngl
        self.nnode = ngl ** dim
        self.nnodedge = ngl - 2
        self.nnodcell = (ngl - 2) ** dim
        self.elemType = 'Spectral{}D(NGL={})'.format(dim, ngl)

        if dim == 2:
            self.indWCurl=[[0,0,1],[1,0,0]]
            self.indCurl=[[0,1,0],[0,0,1]]
            self.indBdiv=[[0,1],[1,2]]
            self.setUpSpectralMats2D(ngl)
        elif dim == 3:
            self.indWCurl=[[0,2,1],[0,1,2],[1,0,2], [1,2,0],[2,1,0],[2,0,1]]
            self.indCurl=[[0,2,1],[0,1,2], [1,0,2], [1,2,0],[2,1,0],[2,0,1]]
            self.indBdiv=[[0,1,5],[1,2,3],[5,3,4]]
            self.nnodface = (ngl - 2)**2
            self.setUpSpectralMats3D(ngl)
        else:
            raise Exception

    def __repr__(self):
        return self.elemType

    def setUpSpectralMats2D(self, ngl):
        nodes1D, operWei = lobattoPoints(ngl)
        gps1D, fullWei = gaussPoints(ngl) if ngl <= 3 else \
            lobattoPoints(ngl)
        gps_red1D, redWei = gaussPoints(ngl - 1)
        cnodes1D, _ = lobattoPoints(2)
        (self.H, self.Hrs, self.gps) = \
            self.computeMats2D(nodes1D, gps1D, fullWei)

        (self.HRed, self.HrsRed, self.gpsRed) = \
            self.computeMats2D(nodes1D, gps_red1D, redWei)

        (self.HOp, self.HrsOp, self.gpsOp) = \
            self.computeMats2D(nodes1D, nodes1D, operWei)

        (self.HCoo, self.HrsCoo, self.gpsCoo) = \
            self.computeMats2D(cnodes1D, gps1D, fullWei)

        (self.HCooRed, self.HrsCooRed, self.gpsCooRed) = \
            self.computeMats2D(cnodes1D, gps_red1D, redWei)

        (self.HCooOp, self.HrsCooOp, self.gpsCooOp) = \
            self.computeMats2D(cnodes1D, nodes1D, operWei)
        (self.HCoo1D, _) = self.interpFun1D(cnodes1D, nodes1D)

    def setUpSpectralMats3D(self, ngl):
        nodes1D, operWei = lobattoPoints(ngl)
        gps1D, fullWei = gaussPoints(ngl) if ngl <= 3 else \
            lobattoPoints(ngl)
        gps_red1D, redWei = gaussPoints(ngl - 1)
        cnodes1D, _ = lobattoPoints(2)
        (self.H, self.Hrs, self.gps) = \
            self.computeMats3D(nodes1D, gps1D, fullWei)

        (self.HRed, self.HrsRed, self.gpsRed) = \
            self.computeMats3D(nodes1D, gps_red1D, redWei)

        (self.HOp, self.HrsOp, self.gpsOp) = \
            self.computeMats3D(nodes1D, nodes1D, operWei)

        (self.HCoo, self.HrsCoo, self.gpsCoo) = \
            self.computeMats3D(cnodes1D, gps1D, fullWei)

        (self.HCooRed, self.HrsCooRed, self.gpsCooRed) = \
            self.computeMats3D(cnodes1D, gps_red1D, redWei)

        (self.HCooOp, self.HrsCooOp, self.gpsCooOp) = \
            self.computeMats3D(cnodes1D, nodes1D, operWei)
        (self.HCoo1D, _) = self.interpFun1D(cnodes1D, nodes1D)

    def getElemKLEMatrices(self, coords):
        """Get the elementary matrices of the KLE Method."""
        # self.logger.debug("getElemKLEMatrices")
        coords.shape = (int(len(coords)/self.dim), self.dim)
        alpha_w = 1e2
        alpha_d = 1e3

        # FIXME Parametrize in terms of total element nodes and for the
        # geometry use a reduced set
        elTotNodes = self.nnode

        elStiffMat = np.zeros((self.dim*elTotNodes,
                                      self.dim*elTotNodes))
        elR_wMat = np.zeros((self.dim*elTotNodes,
                                    self.dim_w*elTotNodes))
        elR_dMat = np.zeros((self.dim*elTotNodes, elTotNodes))

        # Velocity interpolation
        Hvel = np.zeros((self.dim, self.dim*elTotNodes))
        # Velocity gradient
        B_gr = np.zeros((self.dim**2, self.dim*elTotNodes))
        # Velocity divergence
        B_div = np.zeros((1, self.dim*elTotNodes))
        # Velocity curl
        B_curl = np.zeros((self.dim_w, self.dim*elTotNodes))
        # Vorticty curl
        Bw_curl = np.zeros((self.dim, self.dim_w*elTotNodes))

        for idx, gp in enumerate(self.gps):
            Hrs = self.Hrs[idx]
            H = self.H[idx]
            J = self.HrsCoo[idx].dot(coords)
            Hxy = inv(J).dot(Hrs)
            detJ = det(J)

            for nd in range(self.dim):
                B_gr[self.dim*nd:self.dim*nd + self.dim, nd::self.dim] = Hxy
                Hvel[nd, nd::self.dim] = H

            for i,ind in enumerate (self.indWCurl):
                Bw_curl[ind[0],ind[1]::self.dim_w]= (-1)**(i)*Hxy[ind[2]]
            
            elStiffMat += gp.w * detJ * B_gr.T.dot(B_gr)
            elR_wMat += gp.w * detJ * Hvel.T.dot(Bw_curl)
            elR_dMat -= gp.w * detJ * Hvel.T.dot(Hxy)
        
        Hvel = np.zeros((self.dim_w, self.dim_w*elTotNodes))
        # Reduced integration of penalizations
        for idx, gp in enumerate(self.gpsRed):
            Hrs = self.HrsRed[idx]
            H = self.HRed[idx]
            J = self.HrsCooRed[idx].dot(coords)
            Hxy = inv(J).dot(Hrs)
            detJ = det(J)

            for nd in range(self.dim):
                B_div[0, nd::self.dim] = Hxy[nd]
            
            for i,ind in enumerate (self.indCurl):
                B_curl[ind[0],ind[1]::self.dim]= (-1)**(i)*Hxy[ind[2]]
            for nd in range(self.dim_w):
                Hvel[nd, nd::self.dim_w] = H
            
            elStiffMat += gp.w * detJ * (alpha_d * B_div.T.dot(B_div) +
                                         + alpha_w * B_curl.T.dot(B_curl))

            elR_wMat += gp.w * detJ * alpha_w * B_curl.T.dot(Hvel)
            elR_dMat += gp.w * detJ * alpha_d * np.outer(Hxy.flatten("F"), H)
        return (elStiffMat, elR_wMat, elR_dMat)

    def getElemKLEOperators(self, coords):
        coords.shape = (int(len(coords)/self.dim), self.dim)
        elTotNodes = self.nnode
        elSTensorMat = np.zeros((self.dim_s*elTotNodes,
                                        self.dim*elTotNodes))
        # The strain rate tensor is symmetric, thus we work with reduced
        # number of components
        elDivSTMat = np.zeros((self.dim*elTotNodes,
                                      self.dim_s*elTotNodes))
        elCurlMat = np.zeros((self.dim_w*elTotNodes,
                                     self.dim*elTotNodes))
        elWeigMat = np.zeros((elTotNodes, elTotNodes))
        # Strain rate Tensor
        B_srt = np.zeros((self.dim_s, self.dim*elTotNodes))
        Hsrt = np.zeros((self.dim_s, self.dim_s*elTotNodes))
        # Velocity gradient divergence
        B_div = np.zeros((self.dim, self.dim_s*elTotNodes))
        Hdiv = np.zeros((self.dim, self.dim*elTotNodes))
        # Velocity curl
        B_curl = np.zeros((self.dim_w, self.dim*elTotNodes))
        Hcurl = np.zeros((self.dim_w, self.dim_w*elTotNodes))

        for idx, gp in enumerate(self.gpsOp):
            Hrs = self.HrsOp[idx]
            H = self.HOp[idx]

            J = self.HrsCooOp[idx].dot( coords )
            Hxy = inv(J).dot(Hrs)
            detJ = det(J)

            for nd_s in range(self.dim_s):
                Hsrt[nd_s, nd_s::self.dim_s] = H

            for i,ind in enumerate (self.indCurl):
                B_curl[ind[0],ind[1]::self.dim]= (-1)**(i)*Hxy[ind[2]]
            
            #B_srt ONLY FOR INCOMPRESSIBLE, comented the last one
            for x in range(self.dim):
                Hdiv[x, x::self.dim] = H
                B_srt[2*x,x::self.dim]=2*Hxy[x]
                for i in range(self.dim):
                    B_div[i,self.indBdiv[x][i]::self.dim_s]= Hxy[x]
                    ##B_srt[self.indBdiv[x][i],i::self.dim]=Hxy[x]
                    if i!=x :
                        if x+i!=2: 
                            B_srt[x+i,i::self.dim]=Hxy[x]
                        else:
                            B_srt[5,i::self.dim]=Hxy[x]

            ## B_srt[0, 1::self.dim] = -Hxy[1]
            ## B_srt[2, 0::self.dim] = -Hxy[0]
            ## for i in range(self.dim_s-4):
            ##     B_srt[4, i::self.dim] = -Hxy[i]
            ##     B_srt[2*i, 2::self.dim] = -Hxy[2]

            B_srt *= 0.5
            
            for i in range(self.dim_w):
                Hcurl[i,i::self.dim_w] = H

            elSTensorMat += gp.w * detJ * Hsrt.T.dot(B_srt)
            elDivSTMat += gp.w * detJ * Hdiv.T.dot(B_div )
            elCurlMat += gp.w * detJ * Hcurl.T.dot(B_curl)
            elWeigMat += gp.w * detJ * np.outer(H, H)

        elWeigVec = elWeigMat.sum(1)
        return (elSTensorMat, elDivSTMat, elCurlMat, elWeigVec)

    def computeMats2D(self, nodes1D, gps1D, gps1Dwei):
        (h1D, dh1D) = self.interpFun1D(nodes1D, gps1D)
        nNodes = len(nodes1D)
        ngps = len(gps1D)
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
            H.append(np.array([auxRow[y] for y in invPerm]))
            # print("inv",invPerm)
            # print("aux",auxRow)

        # Derivatives of H wrt R & S
        Hrs = list()
        for doubleTern in itertools.product(dh1D, h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs.append(np.array([[auxRow[y] for y in invPerm],
                               [0]*len(invPerm)]))

        for ind, doubleTern in enumerate(itertools.product(h1D, dh1D)):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs[ind][1, :] = [auxRow[y] for y in invPerm]

        gps = generateGaussPoints2D(gps1D, gps1Dwei)
        H = [H[i] for i in invPerm2]
        Hrs = [Hrs[i] for i in invPerm2]
        gps = [gps[i] for i in invPerm2]

        return (H, Hrs, gps)

    def computeMats3D(self, nodes1D, gps1D, gps1Dwei):
        """Interpolate functions in 3D."""
        (h1D, dh1D) = self.interpFun1D(nodes1D, gps1D)
        nNodes = len(nodes1D)
        ngps = len(gps1D)

        invPerm = self.getSpectralOrder(nNodes)
        invPerm2 = self.getSpectralOrder(ngps)

        # Interpolation functions H
        H = list()
        for doubleTern in itertools.product(h1D, h1D ,h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            H.append(np.array([auxRow[y] for y in invPerm]))

        # Derivatives of H wrt  R , S & T
        Hrs = list()
        for doubleTern in itertools.product(dh1D, h1D , h1D):
            auxRow = [reduce(operator.mul, twoPle, 1) for twoPle
                      in itertools.product(*doubleTern)]
            Hrs.append(np.array([[auxRow[y] for y in invPerm],
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
        gps = generateGaussPoints3D(gps1D, gps1Dwei)

        H = [H[i] for i in invPerm2]
        Hrs = [Hrs[i] for i in invPerm2]
        gps = [gps[i] for i in invPerm2]
        
        return (H, Hrs, gps)

    @staticmethod
    def getSpectralOrder(nPoints):
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