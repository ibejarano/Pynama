from petsc4py import PETSc
from matrices.mat_fs import MatFS
import numpy as np

class MatNS(MatFS):
    def preAlloc_Kfs_Krhsfs(self, ind_d, ind_o, globalNodesNS):
        dim = self.__dom.getDimension()
        nodeStart, nodeEnd = self.__dom.getNodesRange()

        locElRow = nodeEnd - nodeStart
        vel_dofs = locElRow * dim

        dns_nnz = np.zeros(locElRow)
        ons_nnz = np.zeros(locElRow)
        drhsns_nnz = np.zeros(locElRow)
        orhsns_nnz = np.zeros(locElRow)

        for node, connect in enumerate(ind_d):
            if (node + nodeStart) not in globalNodesNS:
                dns_nnz[ind] = len(connect & globalNodesNS)
            elif (node + nodeStart) in globalNodesNS:
                # FIXME: len() can be distributed on each set operation
                dns_nnz[node] = len(connect | (connect & globalNodesNS))

            drhsns_nnz[node] = len(connect & globalNodesNS)

        for node, connect in enumerate(ind_o):
            if (node + nodeStart) not in globalNodesNS:
                ons_nnz[node] = len(connect & globalNodesNS)
            elif (node + nodeStart) in globalNodesNS:
                ons_nnz[node] = len(connect | (connect & globalNodesNS))

            orhsns_nnz[node] = len(connect & globalNodesNS)

        drhsns_nnz_ind, orhsns_nnz_ind = self.createNNZWithArray(drhsns_nnz,orhsns_nnz, dim, dim)
        dns_nnz_ind, ons_nnz_ind =self.createNNZWithArray(dns_nnz, ons_nnz, dim, dim)

        self.Kfs = self.createEmptyMat(vel_dofs,vel_dofs,dns_nnz_ind, ons_nnz_ind)
        self.Kfs.setName("Kfs")
        self.kle.append(self.Kfs)

        self.Krhsfs = self.createEmptyMat(vel_dofs,vel_dofs, drhsns_nnz_ind, orhsns_nnz_ind)
        self.Krhsfs.setName("Krhsfs")
        self.kle.append(self.Krhsfs)

    def build(self, buildKLE=True, buildOperators=True):
        locNodesNS = np.array(list(self.__dom.getNodesNoSlip()))
        nodeStart, _ = self.__dom.getNodesRange()
        locNodesNS -= nodeStart
        globNodesNS = self.__dom.getNodesNoSlip(collect=True)

        dim = self.__dom.getDimension()
        locIndNS = [ node*dim+dof for node in locNodesNS for dof in range(dim) ]
        conn_diag, conn_offset, nnz_diag, nnz_off = self.__dom.getConnectivity()

        if buildOperators:
            self.preAlloc_operators(nnz_diag, nnz_off)
        if buildKLE:
            self.preAlloc_Rd_Rw(nnz_diag, nnz_off, locIndNS, createFS=True)
            self.preAlloc_K_Krhs(conn_diag, conn_offset, nnz_diag, nnz_off, locIndNS, globNodesNS)
            self.preAlloc_Kfs_Krhsfs(conn_diag, conn_offset, globNodesNS)

        self.buildNS(globNodesNS)
        self.buildOperators()