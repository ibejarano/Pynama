import numpy as np 
from petsc4py import PETSc

from domain.dmplex_bc import NewBoxDom

class Operators:
    comm = PETSc.COMM_WORLD
    def __init__(self):
        self.Curl = None
        self.Div = None
        self.Srt = None

    def preallocate(self, config, ngl):
        dm = NewBoxDom()
        dm.create(config)
        self.dim = dm.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        assert self.dim_w == 1, "Only for dim = 2"
        
        dm.setFemIndexing(ngl, bc=False, dofs=self.dim_w, fieldName='vorticity')
        conecMat = dm.createMat()
        rStart, rEnd = conecMat.getOwnershipRange()
        locElRow = rEnd - rStart

        nnz_diag = np.zeros(locElRow, dtype=np.int32)
        nnz_off = np.zeros(locElRow, dtype=np.int32)

        for row in range(rStart, rEnd):
            cols, _ = conecMat.getRow(row)
            locRow = row - rStart
            mask_diag = np.logical_and(cols >= rStart,cols < rEnd)
            mask_off = np.logical_or(cols < rStart,cols >= rEnd)
            nnz_diag[locRow] = len(cols[mask_diag])
            nnz_off[locRow] = len(cols[mask_off]) 
        
        conecMat.destroy()
        dm.destroy()

        self.createAll(nnz_diag, nnz_off)

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

    def createEmptyMat(self, rows, cols, d_nonzero, offset_nonzero):
        mat = PETSc.Mat().createAIJ(((rows, None), (cols, None)),
            nnz=(d_nonzero, offset_nonzero), comm=self.comm)
        mat.setUp()
        return mat