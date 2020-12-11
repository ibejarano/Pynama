from petsc4py import PETSc
import numpy as np

class Mat:
    def __init__(self, dim, comm=PETSc.COMM_WORLD):
        self.dim = dim
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        self.comm = comm
        self.mats = list()

    def assembleAll(self):
        for m in self.mats:
            m.assemble()

    def isParallel(self):
        return self.comm.rank > 1

    def getGlobalIndices(self, localIndices):
        globalIndices = set()
        if self.isParallel():
            collectIndices = self.comm.allgather([localIndices])
            for remoteIndices in collectIndices:
                globalIndices |= remoteIndices[0]
        else:
            globalIndices = localIndices
        return globalIndices
        
    def createEmptyKLEMats(self,rStart, rEnd ,  d_nnz_ind , o_nnz_ind, ind_d, ind_o, indicesDIR):
        self.globalIndicesDIR = self.getGlobalIndices(indicesDIR)
        locElRow = rEnd - rStart
        vel_dofs = locElRow * self.dim
        vort_dofs = locElRow * self.dim_w
        # Create matrices for the resolution of the KLE and vorticity transport
        # Create array of NNZ from d_nnz_ind and o_nnz_ind to create Rw
        dw_nnz, ow_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim_w, self.dim)
        # Create array of NNZ from d_nnz_ind and o_nnz_ind to create Rd
        dd_nnz, od_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, 1, self.dim)

        drhs_nnz_ind = np.zeros(locElRow)
        orhs_nnz_ind = np.zeros(locElRow)

        for indRow, indSet in enumerate(ind_d):
            if (indRow + rStart) in self.globalIndicesDIR:
                drhs_nnz_ind[indRow] = 1
            else:
                drhs_nnz_ind[indRow] = len(indSet & self.globalIndicesDIR)
                d_nnz_ind[indRow] = d_nnz_ind[indRow] - len(indSet & self.globalIndicesDIR)
        for indRow, indSet in enumerate(ind_o):
            orhs_nnz_ind[indRow] = len(indSet & self.globalIndicesDIR)

        # Create array of NNZ from d_nnz_ind and o_nnz_ind to create K
        d_nnz, o_nnz = self.createNNZWithArray(d_nnz_ind, o_nnz_ind, self.dim, self.dim )

        # FIXME: This reserves self.dim nonzeros for each node with
        # Dirichlet conditions despite the number of DoF conditioned
        drhs_nnz, orhs_nnz = self.createNNZWithArray(drhs_nnz_ind, orhs_nnz_ind, self.dim, self.dim)
        drhs_nnz[np.array(list(indicesDIR))*2] = 1
        drhs_nnz[np.array(list(indicesDIR))*2+1] = 1

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
        self.K.setName("K")
        self.Rw = self.createEmptyMat(vel_dofs, vort_dofs, dw_nnz, ow_nnz)
        self.Rw.setName("Rw")
        self.Rd = self.createEmptyMat(vel_dofs, locElRow, dd_nnz, od_nnz)
        self.Rd.setName("Rd")
        self.Krhs = self.createEmptyMat(vel_dofs, vel_dofs, drhs_nnz, orhs_nnz)
        self.Krhs.setName("Krhs")

        self.mats = [self.K, self.Rw, self.Rd, self.Krhs]

    def createEmptyMat(self, rows, cols, d_nonzero, offset_nonzero):
        mat = PETSc.Mat().createAIJ(((rows, None), (cols, None)),
            nnz=(d_nonzero, offset_nonzero), comm=self.comm)
        mat.setUp()
        return mat

    def createNonZeroIndex(self, d_nnz, o_nnz, dim1, dim2):
        di_nnz = [x * dim1 for x in d_nnz for d in range(dim2)]
        oi_nnz = [x * dim1 for x in o_nnz for d in range(dim2)]
        return di_nnz, oi_nnz

    def createNNZWithArray(self, d_nnz: np.array, o_nnz: np.array, dim1: int, dim2: int):
        d_nnz = np.array(d_nnz, dtype=np.int32)
        o_nnz = np.array(o_nnz, dtype=np.int32)
        di_nnz_arr = np.repeat(d_nnz*dim1, dim2)
        oi_nnz_arr = np.repeat(o_nnz*dim1, dim2)
        return di_nnz_arr, oi_nnz_arr

    def setIndices2One(self, indices2one):
        for indd in indices2one:
            self.Krhs.setValues(indd, indd, 1, addv=False)
            self.K.setValues(indd, indd, 1, addv=False)
        self.Krhs.assemble()
        self.K.assemble()

    def printMatsInfo(self):
        print(" MATS INFO ")
        print(f"Mat   | Memory Used [B]  | NZ Unneeded")
        print(f"--------------------------------------")
        for m in self.mats:
            info = m.getInfo()
            print(self.formatMatInfo(m.getName(), info))

    @staticmethod
    def formatMatInfo(name, info):
        return f"{name:{5}} | {info['memory']:{16}} | {info['nz_unneeded']:{10}}"
    

class Operators(Mat):
    def createAll(self, rStart, rEnd, d_nnz_ind, o_nnz_ind):
        locElRow = rEnd - rStart
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


if  __name__ == "__main__":
    mat = Mat(2)