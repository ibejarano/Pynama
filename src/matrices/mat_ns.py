from petsc4py import PETSc
from matrices.mat_generator import Mat

class MatNS(Mat):
    def __init__(self, dim, comm):
        self.dim = dim
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        self.comm = comm
        print('noslip!')

    def buildMatrices(self):
        self.buildKLEMatrices()
        self.buildOperatorsMatrices()
        pass

    def buildKLEMatrices(self):
        pass

    def buildOperatorsMatrices(self):
        pass

    def assembleAll(self):
        self.K.assemble()
        self.Rw.assemble()
        self.Rd.assemble()
        self.Krhs.assemble()

    def setfreeFSSetNS(self):
        if dofFreeFSSetNS:  # bool(dof2NSfs):
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

            self.Rdfs.setValues(gldofFreeFSSetNS, indices,
                                locRd[dofFreeFSSetNS, :], addv=True)
        
    def setIndices2One(self, indices2one):
        for indd in indices2one:
            self.Krhs.setValues(indd, indd, 1, addv=False)
            self.K.setValues(indd, indd, 1, addv=False)
        self.Krhs.assemble()
        self.K.assemble()

    def createEmptyKLEMatsNS(self, conecMat, indicesNS, indicesDIR=set(), createOperators=False):
        self.globalIndicesNS =set()
        self.globalIndicesDIR =set()

        # global indices for DIR and NS BC are allgathered among processes
        collectIndices = self.comm.allgather([indicesDIR, indicesNS])
        for remoteIndices in collectIndices:
            self.globalIndicesDIR |= remoteIndices[0] 
            self.globalIndicesNS |= remoteIndices[1]

        # -----
        rStart, rEnd = conecMat.getOwnershipRange()
        locElRow = rEnd - rStart
        vel_dofs = locElRow * self.dim
        vort_dofs = locElRow * self.dim_w
        vtensv_dofs = locElRow * self.dim_s

        ind_d = [0] * (rEnd - rStart)
        ind_o = [0] * (rEnd - rStart)
        for row in range(rStart, rEnd):
            cols, vals = conecMat.getRow(row)
            locRow = row - rStart

            ind_d[locRow] = set([c for c in cols
                                 if ((c >= rStart) and (c < rEnd))])
            ind_o[locRow] = set([c for c in cols
                                 if ((c < rStart) or (c >= rEnd))])
        conecMat.destroy()

        d_nnz_ind = [len(indSet) for indSet in ind_d]
        o_nnz_ind = [len(indSet) for indSet in ind_o]

        # Limit nonzeros in diagonal block row to number of local rows
        locElRow = rEnd - rStart
        d_nnz_ind = [x if x <= locElRow else locElRow for x in d_nnz_ind]

        # Create matrices for the resolution of the KLE and vorticity transport
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create K
        d_nnz, o_nnz = self.createNonZeroIndex(d_nnz_ind, o_nnz_ind, self.dim, self.dim )
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rw
        dw_nnz, ow_nnz = self.createNonZeroIndex(d_nnz_ind, o_nnz_ind, self.dim_w, self.dim)
        # Create list of NNZ from d_nnz_ind and o_nnz_ind to create Rd
        dd_nnz, od_nnz = self.createNonZeroIndex(d_nnz_ind, o_nnz_ind, 1, self.dim)

        drhs_nnz_ind = [0] * (rEnd - rStart)
        orhs_nnz_ind = [0] * (rEnd - rStart)

        for indRow, indSet in enumerate(ind_d):
            if (indRow + rStart) in self.globalIndicesDIR:
                drhs_nnz_ind[indRow] = 1
            else:
                drhs_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR |
                                                     self.globalIndicesNS))
        for indRow, indSet in enumerate(ind_o):
            orhs_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR |
                                                 self.globalIndicesNS))

        # FIXME: This reserves self.dim nonzeros for each node with
        # Dirichlet conditions despite the number of DoF conditioned
        drhs_nnz, orhs_nnz = self.createNonZeroIndex(drhs_nnz_ind, orhs_nnz_ind, self.dim, self.dim)
        
        dns_nnz_ind = [0] * (rEnd - rStart)
        ons_nnz_ind = [0] * (rEnd - rStart)
        for ind, indSet in enumerate(ind_d):
            if (ind + rStart) not in (self.globalIndicesNS |
                                        self.globalIndicesDIR):
                dns_nnz_ind[ind] = len(indSet & self.globalIndicesNS)
            elif (ind + rStart) in self.globalIndicesNS:
                # FIXME: len() can be distributed on each set operation
                dns_nnz_ind[ind] = len((indSet - self.globalIndicesDIR) |
                                        (indSet & self.globalIndicesNS))
        for ind, indSet in enumerate(ind_o):
            if (ind + rStart) not in (self.globalIndicesNS |
                                        self.globalIndicesDIR):
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
            if (indRow + rStart) in (self.globalIndicesDIR -
                                        self.globalIndicesNS):
                drhsns_nnz_ind[indRow] = 1
            else:
                drhsns_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR
                                                | self.globalIndicesNS))
        for indRow, indSet in enumerate(ind_o):
            orhsns_nnz_ind[indRow] = len(indSet & (self.globalIndicesDIR
                                            | self.globalIndicesNS))

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

        # print('dns_nnz: ', dns_nnz)
        self.Kfs = self.createEmptyMat(vel_dofs,vel_dofs,dns_nnz, ons_nnz)
        self.Rwfs =self.createEmptyMat(vel_dofs,vort_dofs, dwns_nnz, owns_nnz)
        self.Rdfs =self.createEmptyMat(vel_dofs,locElRow, ddns_nnz, odns_nnz)
        self.Krhsfs = self.createEmptyMat(vel_dofs,vel_dofs, drhsns_nnz, orhsns_nnz)


        for indRow in set(range(rStart, rEnd)) & set(indicesDIR):
            minInd = (indRow - rStart) * self.dim
            maxInd = (indRow - rStart + 1) * self.dim

            d_nnz[minInd:maxInd] = [1] * self.dim
            o_nnz[minInd:maxInd] = [0] * self.dim

            dw_nnz[minInd:maxInd] = [0] * self.dim
            ow_nnz[minInd:maxInd] = [0] * self.dim

            dw_nnz[minInd:maxInd] = [0] * self.dim
            ow_nnz[minInd:maxInd] = [0] * self.dim


        self.K = self.createEmptyMat(vel_dofs, vel_dofs,d_nnz, o_nnz )
        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz, ow_nnz)
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz, od_nnz)
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz, orhs_nnz)
        
        if createOperators:
            self.createEmptyOperators(d_nnz_ind, o_nnz_ind, locElRow)