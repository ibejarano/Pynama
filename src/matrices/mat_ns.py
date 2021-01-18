from petsc4py import PETSc
from matrices.mat_fs import MatFS
import numpy as np

class MatNS(MatFS):
    bcType ="NS"
    def preAlloc_Kfs_Krhsfs(self, ind_d, ind_o, globalNodesNS):
        dim = self.dom.getDimension()
        nodeStart, nodeEnd = self.dom.getNodesRange()

        locElRow = nodeEnd - nodeStart
        vel_dofs = locElRow * dim

        dns_nnz = np.zeros(locElRow)
        ons_nnz = np.zeros(locElRow)
        drhsns_nnz = np.zeros(locElRow)
        orhsns_nnz = np.zeros(locElRow)

        for node, connect in enumerate(ind_d):
            if (node + nodeStart) not in globalNodesNS:
                dns_nnz[node] = len(connect & globalNodesNS)
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

    def buildNS(self):
        indices2one = set() 
        indices2onefs = set()
        cellStart , cellEnd = self.dom.getLocalCellRange()
        globalTangIndicesNS = self.dom.getTangDofs(collect=True)
        globalNormalIndicesNS = self.dom.getNormalDofs(collect=True)
        
        for cell in range(cellStart, cellEnd):
            nodes , inds , localMats = self.dom.computeLocalKLEMats(cell)
            locK, locRw, locRd = localMats
            indicesVel, indicesW = inds

            indicesVelSet = set(indicesVel)
            normalDofs = globalNormalIndicesNS & indicesVelSet
            tangentialDofs = globalTangIndicesNS & indicesVelSet
            tangentialDofs -= normalDofs

            gldofSetFSNS = list(normalDofs)
            gldofFreeFSSetNS = list(tangentialDofs)
            gldofFree = list(indicesVelSet - normalDofs - tangentialDofs)

            dofFree = [ indicesVel.index(i) for i in gldofFree ]
            locNormalDof = [ indicesVel.index(i) for i in normalDofs ]
            locTangDof = [ indicesVel.index(i) for i in tangentialDofs ]

            dofFreeFSSetNS = locTangDof
            dofSetFSNS = locNormalDof
            dof2beSet = list(set(dofFreeFSSetNS) | set(dofSetFSNS))
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]

            if normalDofs | tangentialDofs:
                self.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.Krhs.setValues(indd, indd, 0, addv=True)
                self.Kfs.setValues(gldofFreeFSSetNS, gldofFree,
                                    locK[np.ix_(dofFreeFSSetNS, dofFree)],
                                    addv=True)

                self.Kfs.setValues(gldofFree, gldofFreeFSSetNS,
                                    locK[np.ix_(dofFree, dofFreeFSSetNS)],
                                    addv=True)

                self.Kfs.setValues(
                    gldofFreeFSSetNS, gldofFreeFSSetNS,
                    locK[np.ix_(dofFreeFSSetNS, dofFreeFSSetNS)],
                    addv=True)

                # Indices where diagonal entries should be reduced by 1
                indices2onefs.update(gldofFreeFSSetNS)

                self.Rwfs.setValues(gldofFreeFSSetNS, indicesW,
                                    locRw[dofFreeFSSetNS, :], addv=True)

                self.Rdfs.setValues(gldofFreeFSSetNS, nodes,
                                    locRd[dofFreeFSSetNS, :], addv=True)
                self.Krhsfs.setValues(
                        gldofFreeFSSetNS, gldofSetFSNS,
                        - locK[np.ix_(dofFreeFSSetNS, dofSetFSNS)], addv=True)
                self.Krhsfs.setValues(
                        gldofFree, gldofSetFSNS,
                        - locK[np.ix_(dofFree, dofSetFSNS)], addv=True)
                for indd in gldofSetFSNS:
                        self.Krhsfs.setValues(indd, indd, 0, addv=True)

            self.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.K.setValues(indd, indd, 0, addv=True)

            self.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)
        
        self.assembleAll()
        self.setIndices2One(indices2one)

        for indd in indices2onefs:
            self.Kfs.setValues(indd, indd, -1, addv=True)

        self.Kfs.assemble()
        self.Rwfs.assemble()
        self.Rdfs.assemble()
        self.Krhsfs.assemble()

        for indd in (indices2one - indices2onefs):
            self.Krhsfs.setValues(indd, indd, 1, addv=False)
        self.Krhsfs.assemble()

        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")


    def build(self, buildKLE=True, buildOperators=True):
        locNodesNS = np.array(list(self.dom.getNodesNoSlip()))
        nodeStart, _ = self.dom.getNodesRange()
        locNodesNS -= nodeStart
        globNodesNS = self.dom.getNodesNoSlip(collect=True)

        dim = self.dom.getDimension()
        locIndNS = [ node*dim+dof for node in locNodesNS for dof in range(dim) ]
        conn_diag, conn_offset, nnz_diag, nnz_off = self.dom.getConnectivity()

        if buildOperators:
            self.preAlloc_operators(nnz_diag, nnz_off)
        if buildKLE:
            self.preAlloc_Rd_Rw(nnz_diag, nnz_off, locIndNS, createFS=True)
            self.preAlloc_K_Krhs(conn_diag, conn_offset, nnz_diag, nnz_off, locIndNS, globNodesNS)
            self.preAlloc_Kfs_Krhsfs(conn_diag, conn_offset, globNodesNS)

        self.buildNS()
        self.buildOperators()
