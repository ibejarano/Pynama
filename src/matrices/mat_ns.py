from petsc4py import PETSc
from matrices.mat_fs import MatFS
import numpy as np

class MatNS(MatFS):
    def assembleAll(self):
        self.K.assemble()
        self.Rw.assemble()
        self.Rd.assemble()
        self.Krhs.assemble()

    def createEmptyKLEMats(self,rStart, rEnd ,  d_nnz_ind , o_nnz_ind, ind_d, ind_o, nodesNS):

        globalNodesNS = self.getGlobalIndices(nodesNS)
        locElRow = rEnd - rStart
        vel_dofs = locElRow * self.dim
        vort_dofs = locElRow * self.dim_w

        # Create matrices for the resolution of the KLE and vorticity transport
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create K
        d_nnz, o_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim, self.dim )
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rw
        dw_nnz, ow_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim_w, self.dim)
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rd
        dd_nnz, od_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, 1, self.dim)

        drhs_nnz_ind = np.zeros(locElRow)
        orhs_nnz_ind = np.zeros(locElRow)

        for indRow, indSet in enumerate(ind_d):
            drhs_nnz_ind[indRow] = len(indSet & globalNodesNS)

        for indRow, indSet in enumerate(ind_o):
            orhs_nnz_ind[indRow] = len(indSet & globalNodesNS)

        # FIXME: This reserves self.dim nonzeros for each node with
        # Dirichlet conditions despite the number of DoF conditioned
        drhs_nnz, orhs_nnz = self.createNNZWithArray(drhs_nnz_ind, orhs_nnz_ind, self.dim, self.dim)
        
        dns_nnz_ind = np.zeros(locElRow)
        ons_nnz_ind = np.zeros(locElRow)

        for ind, indSet in enumerate(ind_d):
            if (ind + rStart) not in globalNodesNS:
                dns_nnz_ind[ind] = len(indSet & globalNodesNS)
            elif (ind + rStart) in globalNodesNS:
                # FIXME: len() can be distributed on each set operation
                dns_nnz_ind[ind] = len(indSet | (indSet & globalNodesNS))
        for ind, indSet in enumerate(ind_o):
            if (ind + rStart) not in globalNodesNS:
                ons_nnz_ind[ind] = len(indSet & globalNodesNS)
            elif (ind + rStart) in globalNodesNS:
                ons_nnz_ind[ind] = len(indSet | (indSet & globalNodesNS))

        dns_nnz, ons_nnz =self.createNNZWithArray(dns_nnz_ind, ons_nnz_ind, self.dim, self.dim)

        dwns_nnz = np.zeros(locElRow * self.dim, dtype=np.int32)
        owns_nnz = np.zeros(locElRow * self.dim, dtype=np.int32)

        ddns_nnz = np.zeros(locElRow * self.dim, dtype=np.int32)
        odns_nnz = np.zeros(locElRow * self.dim, dtype=np.int32)

        drhsns_nnz_ind = np.zeros(locElRow)
        orhsns_nnz_ind = np.zeros(locElRow)
        for indRow, indSet in enumerate(ind_d):
            drhsns_nnz_ind[indRow] = len(indSet & globalNodesNS)
                
        for indRow, indSet in enumerate(ind_o):
                orhsns_nnz_ind[indRow] = len(indSet & globalNodesNS)

        drhsns_nnz, orhsns_nnz = self.createNNZWithArray(drhsns_nnz_ind,orhsns_nnz_ind, self.dim,self.dim)

        for k in set(range(rStart, rEnd)) & set(nodesNS):
            minInd = (k - rStart) * self.dim
            maxInd = (k - rStart + 1) * self.dim

            dwns_nnz[minInd:maxInd] = dw_nnz[minInd:maxInd]
            owns_nnz[minInd:maxInd] = ow_nnz[minInd:maxInd]
            dw_nnz[minInd:maxInd] = [0] * self.dim
            ow_nnz[minInd:maxInd] = [0] * self.dim

            ddns_nnz[minInd:maxInd] = dd_nnz[minInd:maxInd]
            odns_nnz[minInd:maxInd] = od_nnz[minInd:maxInd]
            dd_nnz[minInd:maxInd] = [0] * self.dim
            od_nnz[minInd:maxInd] = [0] * self.dim

        self.Kfs = self.createEmptyMat(vel_dofs,vel_dofs,dns_nnz, ons_nnz)
        self.Rwfs =self.createEmptyMat(vel_dofs,vort_dofs, dwns_nnz, owns_nnz)
        self.Rdfs =self.createEmptyMat(vel_dofs,locElRow, ddns_nnz, odns_nnz)
        
        self.Krhsfs = self.createEmptyMat(vel_dofs,vel_dofs, drhsns_nnz, orhsns_nnz)
        self.K = self.createEmptyMat(vel_dofs, vel_dofs,d_nnz, o_nnz )
        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz, ow_nnz)
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz, od_nnz)
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz, orhs_nnz)

class MatFSNS(MatFS):
    def __init__(self, dim, comm):
        self.dim = dim
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        self.comm = comm

    def assembleAll(self):
        self.K.assemble()
        self.Rw.assemble()
        self.Rd.assemble()
        self.Krhs.assemble()

    def createEmptyKLEMats(self,rStart, rEnd ,  d_nnz_ind , o_nnz_ind, ind_d, ind_o, indicesDIR , indicesNS):
        self.globalIndicesDIR =set()
        self.globalIndicesNS =set()

        collectIndices = self.comm.allgather([indicesDIR, indicesNS])
        for remoteIndices in collectIndices:
            self.globalIndicesDIR |= remoteIndices[0]
            self.globalIndicesNS |= remoteIndices[1]

        locElRow = rEnd - rStart
        vel_dofs = locElRow * self.dim
        vort_dofs = locElRow * self.dim_w

        # Create matrices for the resolution of the KLE and vorticity transport
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create K
        d_nnz, o_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim, self.dim )
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rw
        dw_nnz, ow_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim_w, self.dim)
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rd
        dd_nnz, od_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, 1, self.dim)

        drhs_nnz_ind = np.zeros(locElRow)
        orhs_nnz_ind = np.zeros(locElRow)

        for indRow, indSet in enumerate(ind_d):
            if (indRow + rStart) in self.globalIndicesDIR:
                drhs_nnz_ind[indRow] = 1
            else:
                drhs_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR | self.globalIndicesNS))

        for indRow, indSet in enumerate(ind_o):
                orhs_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR | self.globalIndicesNS))

        # FIXME: This reserves self.dim nonzeros for each node with
        # Dirichlet conditions despite the number of DoF conditioned
        drhs_nnz, orhs_nnz = self.createNonZeroIndex(drhs_nnz_ind, orhs_nnz_ind, self.dim, self.dim)
        
        dns_nnz_ind = [0] * (rEnd - rStart)
        ons_nnz_ind = [0] * (rEnd - rStart)
        for ind, indSet in enumerate(ind_d):
            if (ind + rStart) not in (self.globalIndicesNS | self.globalIndicesDIR ):
                dns_nnz_ind[ind] = len(indSet & self.globalIndicesNS)
            elif (ind + rStart) in self.globalIndicesNS:
                # FIXME: len() can be distributed on each set operation
                dns_nnz_ind[ind] = len((indSet - self.globalIndicesDIR) | 
                                        (indSet & self.globalIndicesNS))
        for ind, indSet in enumerate(ind_o):
            if (ind + rStart) not in (self.globalIndicesNS | self.globalIndicesDIR):
                ons_nnz_ind[ind] = len(indSet & self.globalIndicesNS)
            elif (ind + rStart) in self.globalIndicesNS:
                ons_nnz_ind[ind] = len((indSet - self.globalIndicesDIR) | 
                                        (indSet & self.globalIndicesNS))

        dns_nnz, ons_nnz =self.createNonZeroIndex (dns_nnz_ind, ons_nnz_ind, self.dim, self.dim)

        dwns_nnz = [0] * (rEnd - rStart) * self.dim
        owns_nnz = [0] * (rEnd - rStart) * self.dim

        ddns_nnz = [0] * (rEnd - rStart) * self.dim
        odns_nnz = [0] * (rEnd - rStart) * self.dim

        drhsns_nnz_ind = [0] * (rEnd - rStart)
        orhsns_nnz_ind = [0] * (rEnd - rStart)
        for indRow, indSet in enumerate(ind_d):
            if (indRow + rStart) in (self.globalIndicesDIR - self.globalIndicesNS):
                drhsns_nnz_ind[indRow] = 1
            else:
                drhsns_nnz_ind[indRow] = len(indSet & (self.globalIndicesNS | self.globalIndicesDIR))
                
        for indRow, indSet in enumerate(ind_o):
                orhsns_nnz_ind[indRow] = len(indSet & (self.globalIndicesNS | self.globalIndicesDIR))

        drhsns_nnz, orhsns_nnz = self.createNonZeroIndex(drhsns_nnz_ind,orhsns_nnz_ind, self.dim,self.dim)

        # k numbers nodes
        for k in set(range(rStart, rEnd)) & set(indicesNS):
            minInd = (k - rStart) * self.dim
            maxInd = (k - rStart + 1) * self.dim

            dwns_nnz[minInd:maxInd] = dw_nnz[minInd:maxInd]
            owns_nnz[minInd:maxInd] = ow_nnz[minInd:maxInd]
            dw_nnz[minInd:maxInd] = [0] * self.dim
            ow_nnz[minInd:maxInd] = [0] * self.dim

            ddns_nnz[minInd:maxInd] = dd_nnz[minInd:maxInd]
            odns_nnz[minInd:maxInd] = od_nnz[minInd:maxInd]
            dd_nnz[minInd:maxInd] = [0] * self.dim
            od_nnz[minInd:maxInd] = [0] * self.dim

        self.Kfs = self.createEmptyMat(vel_dofs,vel_dofs,dns_nnz, ons_nnz)
        self.Rwfs =self.createEmptyMat(vel_dofs,vort_dofs, dwns_nnz, owns_nnz)
        self.Rdfs =self.createEmptyMat(vel_dofs,locElRow, ddns_nnz, odns_nnz)
        self.Krhsfs = self.createEmptyMat(vel_dofs,vel_dofs, drhsns_nnz, orhsns_nnz)
        # print("RW NNZ", dw_nnz)
        # print("K NNZ", d_nnz)
        self.K = self.createEmptyMat(vel_dofs, vel_dofs,d_nnz, o_nnz )
        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz, ow_nnz)
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz, od_nnz)
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz, orhs_nnz)