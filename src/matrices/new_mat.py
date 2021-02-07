from petsc4py import PETSc
import numpy as np
import logging

from utils.dm_spectral import reorderEntities, getLocalDofsFromCell, getGlobalDofsFromCell
from utils.dm import getTotalCells, getCellCornersCoords

class Matrices:
    def __init__(self):
        self.logger = logging.getLogger("Matrices")

    def setDM(self, dm):
        self.__dm = dm

    def setElem(self, elem):
        self.__elem = elem

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
            points = reorderEntities(points)
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
        ngl = self.__elem.ngl
        dim = dm.getDimension()
        dim_w = 1
        totalDofsFree = dm.getGlobalVec().getSize()
        locVel = dm.getLocalVec()
        totalDofs = locVel.getSize()
        dm.restoreLocalVec(locVel)
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

    def assembleKLEMatrices(self):
        dm = self.__dm
        countCells = 0
        K = dm.createMat()
        Rw = self.preallocRw()
        dim = dm.getDimension()
        assert dim == 2, "Not implemented for dim = 3"
        dim_w = 1
        borderCells = dm.getStratumIS('boundary', 1)
        assert dm.getDimension() == 2, "Check if celltype = 4 in dim = 3"
        allCells = dm.getStratumIS('celltype', 4)
        insideCells = allCells.difference(borderCells)
        lgmap = dm.getLGMap()

        startCell, _ = dm.getHeightStratum(0)

        for cell in insideCells.getIndices():
            cellCornerCoords = getCellCornersCoords(dm, startCell, cell )
            locK, locRw, _ = self.__elem.getElemKLEMatrices(cellCornerCoords)
            velDofs = getGlobalDofsFromCell(dm, cell)
            allVels = getLocalDofsFromCell(dm, cell)
            toGl = lgmap.applyInverse(velDofs)
            velDofsElem = [np.where( allVels == i )[0][0] for i in toGl ]
            
            K.setValues(velDofs, velDofs,
                        locK[np.ix_(velDofsElem, velDofsElem)], addv=True)

            vortDofs = [ int(dof*dim_w/dim) for dof in allVels[::dim]]
            vortDofsElem = list(range(len(vortDofs)))
            
            Rw.setValues(velDofs, vortDofs,
                       locRw[np.ix_(velDofsElem, vortDofsElem)], addv=True)

            countCells += 1
            self.printProgress(countCells)

        bcPoints = dm.getStratumIS('marker', 1).getIndices()
        
        bcDofs = np.zeros(0)
        for poi in bcPoints:
            arrtmp = np.arange(*dm.getPointLocal(poi)).astype(np.int32)
            bcDofs = np.append(bcDofs, arrtmp).astype(np.int32)
        
        Krhs = self.preallocKrhs()
        bcDofs = set(bcDofs)
        for cell in borderCells.getIndices():
            glVelIndices = getLocalDofsFromCell(dm, cell)
            glVelConstraint = list(set(glVelIndices) & bcDofs)
            glVelDofs = list(set(glVelIndices) - bcDofs)

            elemDofs = [np.where( glVelIndices == i )[0][0] for i in glVelDofs ]
            elemConstraint = [np.where( glVelIndices == i )[0][0] for i in glVelConstraint ]
           
            cellCornerCoords = getCellCornersCoords(dm, startCell, cell )
            locK, locRw, _ = self.__elem.getElemKLEMatrices(cellCornerCoords)

            K.setValuesLocal(glVelDofs, glVelDofs, locK[np.ix_(elemDofs, elemDofs)], addv=True)

            localDofs = lgmap.apply(glVelDofs)

            Krhs.setValues(localDofs, glVelConstraint,-locK[np.ix_(elemDofs, elemConstraint)], addv=True )

            vortIndices = [ int(dof*dim_w/dim) for dof in glVelIndices[::dim]]
            vortDofsElem = list(range(len(vortIndices)))
            Rw.setValues(localDofs, vortIndices,
                       locRw[np.ix_(elemDofs, vortDofsElem)], addv=True)
            countCells += 1
            self.printProgress(countCells)

        K.assemble()
        Krhs.assemble()
        Rw.assemble()
        print(" Matrices ensambladas!")
        return K, Krhs, Rw

    def printProgress(self, currCell, width=50):
        totCells = getTotalCells(self.__dm)
        percent = int(currCell*100 / totCells)

        left = width * percent // 100
        right = width - left

        print('\r[', '#' * left, '-' * right, ']',
            f'{currCell}/{totCells} cells',
            sep='', end='', flush=True)