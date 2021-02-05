import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import logging
from math import pi, floor

from utils.dm import getCellCornersCoords
from utils.dm_spectral import getLocalDofsFromCell

class DMPlexDom(PETSc.DMPlex):
    comm = PETSc.COMM_WORLD
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"[{self.comm.rank}] Class")
        self.logger.info("DMPlex Instance Created")

    def create(self):
        self.distribute()
        self.dim = self.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

        if self.dim == 2:
            self.namingConvention = ["down", "right" , "up", "left"]
            self.reorderEntities = lambda entities: np.hstack((entities[5:], entities[1:5], entities[0]))
        elif self.dim == 3:
            self.namingConvention = ["back", "front", "down", "up", "right", "left"]
            self.reorderEntities = lambda entities: np.hstack((entities[19:], entities[7:19], entities[1:7], entities[0]))

        self.markBoundaryFaces('boundary', 0)
        faces = self.getStratumIS('boundary', 0)
        for f in faces.getIndices():
            cell = self.getSupport(f)
            self.setLabelValue('boundary', cell, 1)

    def setFemIndexing(self, ngl,bc=True, dofs=None, fieldName='velocity'):
        fields = 1
        self.__ngl = ngl

        self.setNumFields(fields)
        numComp = [1]
        dofs = dofs if dofs != None else self.getDimension()

        numDofVel = [ 1*dofs , dofs * (ngl-2) , dofs * (ngl-2)**2 ]

        if bc:
            bcIs = self.getStratumIS('marker', 1)
            velSec = self.createSection(numComp, numDofVel, 0, bcPoints=[bcIs])
        else:
            velSec = self.createSection(numComp, numDofVel)
        
        velSec.setFieldName(0, fieldName)
        self.setDefaultSection(velSec)
        self.velSec = self.getDefaultGlobalSection()

    def getNGL(self):
        return self.__ngl

    def getLocalVelocityDofsFromCell(self, cell):
        points, oris = self.getTransitiveClosure(cell)
        arr = np.zeros(0, dtype=np.int32)
        points = self.reorderEntities(points)
        oris = self.reorderEntities(oris)
        for i, poi in enumerate(points):
            arrtmp = np.arange(*self.getPointLocal(poi))
            if oris[i] == -2:
                tmp = arrtmp.copy()
                tmp[-2::-2] = arrtmp[::2]
                tmp[::-2] = arrtmp[1::2]
                arrtmp = tmp
            arr = np.append(arr, arrtmp)
        return arr

    def getGlobalVelocityDofsFromCell(self, cell):
        points, _ = self.getTransitiveClosure(cell)
        arr = np.zeros(0, dtype=np.int32)
        points = self.reorderEntities(points)
        # self.setDefaultGlobalSection(self.velSec)
        for poi in points:
            arrtmp = np.arange(*self.getPointGlobal(poi))
            arr = np.append(arr, arrtmp)
        return arr

    def getGlobalVorticityDofsFromCell(self, cell):
        points, _ = self.getTransitiveClosure(cell)
        arr = np.zeros(0)
        points = self.reorderEntities(points)
        for poi in points:
            arrtmp = np.arange(*self.getPointGlobal(poi))
            arr = np.append(arr, arrtmp)
        return arr.astype(np.int32)

    def computeFullCoordinates(self, spElem):
        dim = self.getDimension()
        fullCoordVec = self.createLocalVec()
        fullCoordVec.setName('Coordinates')

        coordsComponents = dim
        startCell, _ = self.getHeightStratum(0)

        for cell in range(*self.getHeightStratum(0)):
            coords = getCellCornersCoords(self, startCell ,cell)
            coords.shape = (2** coordsComponents , coordsComponents)
            indices = getLocalDofsFromCell(self, cell)
            elTotNodes = spElem.nnode
            totCoord = np.zeros((coordsComponents*elTotNodes))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx]@coords).T

            fullCoordVec.setValues(indices.astype(np.int32), totCoord)
        fullCoordVec.assemble()
        return fullCoordVec

    def computeLocalMatrices(self, cellNum):
        cornerCoords = self.getCellCornersCoords(cellNum)
        return self.__elem.getElemKLEMatrices(cornerCoords)

    def createVortLGMap(self):
        assert self.getDimension() == 2, "Not implemented for dim 3"
        velLGMap = self.getLGMap()
        arr = velLGMap.getBlockIndices()
        lg = PETSc.LGMap().create(arr)
        return lg

class NewBoxDom(DMPlexDom):
    """Estrucuted DMPlex Mesh"""
    def create(self, data):
        lower = data['lower']
        upper = data['upper']
        faces = data['nelem']
        self.createBoxMesh(faces=faces, lower=lower, upper=upper, simplex=False)
        self.logger.info("Box mesh generated")
        super().create()

class NewGmshDom(DMPlexDom):
    """Unstructured DMPlex Mesh"""
    def create(self, fileName: str):
        self.createFromFile(fileName)
        self.logger.info("Mesh generated from Gmsh file")
        super().create()

if __name__ == "__main__":
    data = {"ngl":2, "box-mesh": {
        "nelem": [2,2],
        "lower": [0,0],
        "upper": [1,1]
    }}

    testData = {
        "free-slip": {
            "up": [1, 0],
            "right": [1, 0]},
        "no-slip": {
            "left": [1, 1],
            "down": [None, 0]
        }
    }

    domain = Domain(data)
    domain.newBCSETUP(testData)
    # domain.view()
