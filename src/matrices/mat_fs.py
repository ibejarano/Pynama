from petsc4py import PETSc
import numpy as np
import logging

class MatFS:
    comm = PETSc.COMM_WORLD
    def __init__(self):
        self.logger = logging.getLogger(f"[{self.comm.rank}]:MatClass")
        self.kle = list()
        self.operators = list()

        self.__dom = None

    def setDomain(self, dom):
        self.__dom = dom

    def assembleAll(self):
        for m in self.kle:
            m.assemble()
            self.logger.debug(f"Mat {m.getName()} Assembled")
        
    def createEmptyKLEMats(self,rStart, rEnd ,  d_nnz_ind , o_nnz_ind, ind_d, ind_o, nodesDir):
        globalNodesDir = self.getGlobalIndices(nodesDir)
        locElRow = rEnd - rStart
        vel_dofs = locElRow * self.dim
        vort_dofs = locElRow * self.dim_w
        # Create array of NNZ from d_nnz_ind and o_nnz_ind to create Rw
        dw_nnz, ow_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim_w, self.dim)
        # Create array of NNZ from d_nnz_ind and o_nnz_ind to create Rd
        dd_nnz, od_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, 1, self.dim)

        drhs_nnz_ind = np.zeros(locElRow)
        orhs_nnz_ind = np.zeros(locElRow)

        for indRow, indSet in enumerate(ind_d):
            if (indRow + rStart) in globalNodesDir:
                drhs_nnz_ind[indRow] = 1
            else:
                drhs_nnz_ind[indRow] = len(indSet & globalNodesDir)
                d_nnz_ind[indRow] = d_nnz_ind[indRow] - len(indSet & globalNodesDir)
                
        for indRow, indSet in enumerate(ind_o):
            orhs_nnz_ind[indRow] = len(indSet & globalNodesDir)

        d_nnz, o_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim, self.dim )
        drhs_nnz, orhs_nnz = self.createNNZWithArray(drhs_nnz_ind, orhs_nnz_ind, self.dim, self.dim)

        indicesDIR = [ node*self.dim-(rStart*self.dim) + dof for node in nodesDir for dof in range(self.dim) ]

        d_nnz[indicesDIR] = 1
        o_nnz[indicesDIR] = 0

        dw_nnz[indicesDIR] = 0
        ow_nnz[indicesDIR] = 0

        dw_nnz[indicesDIR] = 0
        ow_nnz[indicesDIR] = 0

        self.K = self.createEmptyMat(vel_dofs, vel_dofs,d_nnz, o_nnz )
        self.K.setName("K")
        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz, ow_nnz)
        self.Rw.setName("Rw")
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz, od_nnz)
        self.Rd.setName("Rd")
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz, orhs_nnz)
        self.Krhs.setName("Krhs")
        self.mats = [self.K, self.Rw, self.Rd, self.Krhs]

    def preAlloc_K_Krhs(self, ind_d, ind_o, d_nnz, o_nnz, locIndicesDir ,globalNodesDir):
        dim = self.__dom.getDimension()
        nodeStart, nodeEnd = self.__dom.getNodesRange()

        locElRow = nodeEnd - nodeStart
        vel_dofs = locElRow * dim

        drhs_nnz = np.zeros(locElRow)
        orhs_nnz = np.zeros(locElRow)

        for node, connectivity in enumerate(ind_d):
            if (node + nodeStart) in globalNodesDir:
                drhs_nnz[node] = 1
            else:
                drhs_nnz[node] = len(connectivity & globalNodesDir)
                d_nnz[node] = d_nnz[node] - len(connectivity & globalNodesDir)
                
        for node, connectivity in enumerate(ind_o):
            orhs_nnz[node] = len(connectivity & globalNodesDir)

        d_nnz_ind, o_nnz_ind = self.createNNZWithArray(d_nnz, o_nnz, dim, dim )
        drhs_nnz_ind, orhs_nnz_ind = self.createNNZWithArray(drhs_nnz, orhs_nnz, dim, dim )

        d_nnz_ind[locIndicesDir] = 1
        o_nnz_ind[locIndicesDir] = 0

        self.K = self.createEmptyMat(vel_dofs, vel_dofs, d_nnz_ind, o_nnz_ind)
        self.K.setName("K")
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz_ind, orhs_nnz_ind)
        self.Krhs.setName("Krhs")
        self.kle.append(self.K)
        self.kle.append(self.Krhs)

    def preAlloc_Rd_Rw(self, diag_nnz, off_nnz, locIndicesDir, createFS=False):
        dim, dim_w, _ = self.__dom.getDimensions()

        locElRow = len(diag_nnz)
        vel_dofs = locElRow * dim
        vort_dofs = locElRow * dim_w

        dw_nnz_ind, ow_nnz_ind = self.createNNZWithArray(diag_nnz, off_nnz, dim_w, dim)
        dd_nnz_ind, od_nnz_ind = self.createNNZWithArray(diag_nnz, off_nnz, 1, dim)


        if createFS:
            dwns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)
            owns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)

            ddns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)
            odns_nnz_ind = np.zeros(locElRow * dim, dtype=np.int32)

            dwns_nnz_ind = dw_nnz_ind[locIndicesDir]
            owns_nnz_ind = ow_nnz_ind[locIndicesDir]

            ddns_nnz_ind = dd_nnz_ind[locIndicesDir]
            odns_nnz_ind = od_nnz_ind[locIndicesDir]

            self.Rwfs =self.createEmptyMat(vel_dofs,vort_dofs, dwns_nnz_ind, owns_nnz_ind)
            self.Rwfs.setName("Rwfs")
            self.kle.append(self.Rwfs)
            self.Rdfs =self.createEmptyMat(vel_dofs,locElRow, ddns_nnz_ind, odns_nnz_ind)
            self.Rdfs.setName("Rdfs")
            self.kle.append(self.Rdfs)

        dw_nnz_ind[locIndicesDir] = 0
        ow_nnz_ind[locIndicesDir] = 0

        dd_nnz_ind[locIndicesDir] = 0
        od_nnz_ind[locIndicesDir] = 0

        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz_ind, ow_nnz_ind)
        self.Rw.setName("Rw")
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz_ind, od_nnz_ind)
        self.Rd.setName("Rd")
        self.kle.append(self.Rw)
        self.kle.append(self.Rd)

    def preAlloc_operators(self, nnz_diag, nnz_off):
        self.operator = Operators()
        dims = self.__dom.getDimensions()
        self.operator.setDimensions(dims)
        self.operator.createAll(nnz_diag, nnz_off)

    def createEmptyMat(self, rows, cols, d_nonzero, offset_nonzero):
        mat = PETSc.Mat().createAIJ(((rows, None), (cols, None)),
            nnz=(d_nonzero, offset_nonzero), comm=self.comm)
        mat.setUp()
        return mat

    def createNNZWithArray(self, d_nnz: np.array, o_nnz: np.array, dim1: int, dim2: int):
        d_nnz = np.array(d_nnz, dtype=np.int32)
        o_nnz = np.array(o_nnz, dtype=np.int32)
        di_nnz_arr = np.repeat(d_nnz*dim1, dim2)
        oi_nnz_arr = np.repeat(o_nnz*dim1, dim2)
        return di_nnz_arr, oi_nnz_arr

    def setIndices2One(self, indices2one):
        for indd in indices2one:
            self.Krhs.setValues(indd, indd, 1, addv=True)
            self.K.setValues(indd, indd, 1, addv=True)
        self.Krhs.assemble()
        self.K.assemble()

    def printMatsInfo(self):
        print(" MATS INFO ")
        print(f"Mat   | Memory Used [B]  | NZ Unneeded")
        print(f"--------------------------------------")
        for m in self.mats:
            info = m.getInfo()
            print(self.formatMatInfo(m.getName(), info))

    def build(self, buildKLE=True, buildOperators=True):
        locNodesDirichlet = np.array(list(self.__dom.getNodesDirichlet()))
        nodeStart, _ = self.__dom.getNodesRange()
        locNodesDirichlet -= nodeStart
        globNodesDirichlet = self.__dom.getNodesDirichlet(collect=True)

        dim = self.__dom.getDimension()
        locIndDirichlet = [ node*dim+dof for node in locNodesDirichlet for dof in range(dim) ]
        conn_diag, conn_offset, nnz_diag, nnz_off = self.__dom.getConnectivity()

        if buildOperators:
            self.preAlloc_operators(nnz_diag, nnz_off)
        if buildKLE:
            self.preAlloc_Rd_Rw(nnz_diag, nnz_off, locIndDirichlet)
            self.preAlloc_K_Krhs(conn_diag, conn_offset, nnz_diag, nnz_off, locIndDirichlet, globNodesDirichlet)

        self.buildFS(globNodesDirichlet)
        self.buildOperators()

    def buildFS(self, globNodesDirichlet):
        cellStart , cellEnd = self.__dom.getLocalCellRange()
        dim = self.__dom.getDimension()
        for cell in range(cellStart, cellEnd):
            nodes , inds , localMats = self.__dom.computeLocalKLEMats(cell)
            locK, locRw, _ = localMats
            indicesVel, indicesW = inds
            
            nodeBCintersect = set(globNodesDirichlet) & set(nodes)
            dofSetFSNS = set()

            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                # FIXME : No importa el bc, #TODO cuando agregemos NS si importa
                for dof in range(dim):
                    dofSetFSNS.add(localBoundaryNode*dim + dof)

            dofFree = list(set(range(len(indicesVel)))
                            - dofSetFSNS)
            dof2beSet = list(dofSetFSNS)
            dofSetFSNS = list(dofSetFSNS)
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            
            if nodeBCintersect:
                self.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)

            self.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.K.setValues(indd, indd, 0, addv=True)

            self.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

        globalIndicesDIR = [node*dim + dof for node in globNodesDirichlet for dof in range(dim) ] 
        self.setIndices2One(globalIndicesDIR)
        self.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")

    def buildOperators(self):
        cellStart, cellEnd = self.__dom.getLocalCellRange()
        for cell in range(cellStart, cellEnd):
            nodes, localOperators = self.__dom.computeLocalOperators(cell)
            self.operator.setValues(localOperators, nodes)
        self.operator.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"Operators Matrices builded")

    def getOperators(self):
        return self.operator

    @staticmethod
    def formatMatInfo(name, info):
        return f"{name:{5}} | {info['memory']:{16}} | {info['nz_unneeded']:{10}}"
    

class Operators(MatFS):
    def setDimensions(self, dims):
        self.dim, self.dim_w, self.dim_s = dims

    def createAll(self, d_nnz_ind, o_nnz_ind):
        locElRow = len(d_nnz_ind)
        self.createCurl(d_nnz_ind, o_nnz_ind,locElRow)
        self.createDivSrt(d_nnz_ind, o_nnz_ind,locElRow)
        self.createSrT(d_nnz_ind, o_nnz_ind,locElRow)
    
    def create(self,dim1,dim2, d_nnz_ind, o_nnz_ind, locElRow):
        d_nnz = [x * dim1 for x in d_nnz_ind for d in range(dim2)]
        o_nnz = [x * dim1 for x in o_nnz_ind for d in range(dim2)]
        return self.createEmptyMat(locElRow * dim2 ,locElRow * dim1 ,d_nnz, o_nnz)

    def createCurl(self, d_nnz_ind, o_nnz_ind,locElRow):
        self.Curl = self.create(self.dim,self.dim_w, d_nnz_ind, o_nnz_ind, locElRow)
        self.weigCurl = PETSc.Vec().createMPI(((locElRow * self.dim_w, None)),comm=self.comm)

    def createDivSrt(self, d_nnz_ind, o_nnz_ind,locElRow):
        self.DivSrT = self.create(self.dim_s,self.dim, d_nnz_ind, o_nnz_ind, locElRow)
        self.weigDivSrT = PETSc.Vec().createMPI(((locElRow * self.dim, None)), comm=self.comm)

    def createSrT(self, d_nnz_ind, o_nnz_ind,locElRow):
        self.SrT = self.create(self.dim, self.dim_s, d_nnz_ind, o_nnz_ind, locElRow)
        self.weigSrT = PETSc.Vec().createMPI(((locElRow * self.dim_s, None)), comm=self.comm)
    
    def setValues(self, localOperators, nodes):
        locSrT, locDivSrT, locCurl, locWei = localOperators

        indicesVel = [node*self.dim + dof for node in nodes for dof in range(self.dim)]
        indicesW = [node*self.dim_w + dof for node in nodes for dof in range(self.dim_w)]
        indicesSrT = [node*self.dim_s + dof for node in nodes for dof in range(self.dim_s)]

        self.Curl.setValues(indicesW, indicesVel, locCurl, True)
        self.SrT.setValues(indicesSrT, indicesVel, locSrT, True)
        self.DivSrT.setValues(indicesVel, indicesSrT, locDivSrT, True)

        self.weigSrT.setValues(indicesSrT, np.repeat(locWei, self.dim_s), True)
        self.weigDivSrT.setValues(indicesVel, np.repeat(locWei, self.dim), True)
        self.weigCurl.setValues(indicesW, np.repeat(locWei, self.dim_w), True)

    def assembleAll(self):
        self.SrT.assemble()
        self.weigSrT.assemble()
        self.weigSrT.reciprocal()
        self.SrT.diagonalScale(L=self.weigSrT)

        self.DivSrT.assemble()
        self.weigDivSrT.assemble()
        self.weigDivSrT.reciprocal()
        self.DivSrT.diagonalScale(L=self.weigDivSrT)

        self.Curl.assemble()
        self.weigCurl.assemble()
        self.weigCurl.reciprocal()
        self.Curl.diagonalScale(L=self.weigCurl)

        self.weigSrT.destroy()
        self.weigCurl.destroy()
        self.weigDivSrT.destroy()