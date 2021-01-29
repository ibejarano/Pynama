from petsc4py import PETSc
import numpy as np
import logging

class Matrices:
    def __init__(self):
        self.logger = logging.getLogger("Matrices")

    def setDM(self, dm):
        self.__dm = dm

    def preallocKrhs(self):
        dm = self.__dm

        bcPoints = set(dm.getStratumIS('marker', 1).getIndices())
        bcCells = dm.getStratumIS('boundary', 1)
        
        sec = dm.getDefaultSection()
        maxDofs = sec.getConstrainedStorageSize()
        maxCols = sec.getStorageSize()
        d_nnz = np.zeros(maxDofs, dtype=np.int32)
        
        for cell in bcCells.getIndices():
            points = dm.getTransitiveClosure(cell)[0]
            points = dm.reorderEntities(points)
            bcCellPoints = set(points) & bcPoints
            bcDofs = 0
            for poi in bcCellPoints:
                dofs = sec.getConstraintDof(poi)
                bcDofs += dofs
            
            arr = np.zeros(0, dtype=np.int32)
            for poi in points:
                arrtmp = np.arange(*dm.getPointGlobal(poi))
                arr = np.append(arr, arrtmp)
                
            d_nnz[arr] += bcDofs
            
        Krhs = PETSc.Mat().createAIJ(((maxDofs, None), (maxCols, None)),
                nnz=(d_nnz, 0), comm=dm.comm)
        Krhs.setUp()
        return Krhs

    def preallocRw(self):
        dm = self.__dm

        poiTypes = (4, 1, 0)
        ngl = dm.getNGL()
        dim = dm.getDimension()
        dim_w = 1
        totalDofsFree = dm.getGlobalVec().getSize()
        totalDofs = dm.getLocalVec().getSize()
        dofsPerVertex = 1
        dofsPerEdge = ngl - 2
        dofsPerCell = (ngl-2)**2
        dofsPerType = ( 4*dofsPerEdge + dofsPerCell + 4*dofsPerVertex , 
                    7*dofsPerEdge + 2*dofsPerCell + 6*dofsPerVertex  , 
                    12*dofsPerEdge + 4*dofsPerCell + 9*dofsPerVertex)
        d_nnz = np.zeros(totalDofsFree, dtype=np.int32)
        for i, poiType in enumerate(poiTypes):
            pois = dm.getStratumIS("celltype", poiType)
            arr = np.zeros(0)
            for poi in pois.getIndices():
                arrtmp = np.arange(*dm.getPointGlobal(poi), dtype=np.int32)
                arr = np.append(arr, arrtmp).astype(np.int32)
            d_nnz[arr] = dim_w * dofsPerType[i]
        cols = int(totalDofs*dim_w/dim)
        mat = PETSc.Mat().createAIJ(size=(totalDofsFree, cols),nnz=(d_nnz, 0))
        return mat

    def assembleK(self):
        dm = self.__dm

        K = dm.createMat()
        borderCells = dm.getStratumIS('boundary', 1)
        assert dm.getDimension() == 2, "Check if celltype = 4 in dim = 3"
        allCells = dm.getStratumIS('celltype', 4)
        insideCells = allCells.difference(borderCells)
        for cell in insideCells.getIndices():
            # Get jacobian insted of this
            locK, _, _ = dm.computeLocalMatrices(cell)
            velDofs = dm.getGlobalVelocityDofsFromCell(cell).astype(np.int32)
            velDofsElem = list(range(len(velDofs)))
            
            K.setValues(velDofs, velDofs,
                        locK[np.ix_(velDofsElem, velDofsElem)], addv=True)

        bcPoints = dm.getStratumIS('marker', 1).getIndices()
        
        bcDofs = np.zeros(0)
        for poi in bcPoints:
            arrtmp = np.arange(*dm.getPointLocal(poi)).astype(np.int32)
            bcDofs = np.append(bcDofs, arrtmp).astype(np.int32)
        
        Krhs = self.preallocKrhs()

        bcDofs = set(bcDofs)
        
        for cell in borderCells.getIndices():
            velDofs = dm.getLocalVelocityDofsFromCell(cell)
            localDofsConstraint = list(set(velDofs) & bcDofs)
            localDofs = list(set(velDofs) - bcDofs)
            elemDofs = [np.where( velDofs == i )[0][0] for i in localDofs ]
            elemConstraint = [np.where( velDofs == i )[0][0] for i in localDofsConstraint ]
           
            locK, _, _ = dm.computeLocalMatrices(cell)
            
            K.setValuesLocal(localDofs, localDofs, locK[np.ix_(elemDofs, elemDofs)], addv=True)
            lgmap = K.getLGMap()[0]
            glDofs = lgmap.apply(localDofs)
            Krhs.setValues(glDofs, localDofsConstraint,-locK[np.ix_(elemDofs, elemConstraint)], addv=True )
        K.assemble()
        Krhs.assemble()
        return K, Krhs


    def assembleRw(self):
        Rw = self.preallocRw()
        dm = self.__dm
        dim = dm.getDimension()
        assert dim == 2, "Not implemented for dim = 3"
        dim_w = 1
        lgmap = dm.getLGMap()
        for cell in range(*dm.getCellRange()):
            _ , locRw , _ = dm.computeLocalMatrices(cell)
            velDofs = dm.getGlobalVelocityDofsFromCell(cell).astype(np.int32)
            allVels = dm.getLocalVelocityDofsFromCell(cell).astype(np.int32)

            toGl = lgmap.applyInverse(velDofs)
            velDofsElem = [np.where( allVels == i )[0][0] for i in toGl ]

            vortDofs = [ int(dof*dim_w/dim) for dof in allVels[::dim]]
            vortDofsElem = list(range(len(vortDofs)))
            
            Rw.setValues(velDofs, vortDofs,
                        locRw[np.ix_(velDofsElem, vortDofsElem)], addv=True)

        Rw.assemble()
        return Rw