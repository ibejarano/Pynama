import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import logging
from math import pi, floor
from .elements.spectral import Spectral2

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

    def setFemIndexing(self, ngl):
        fields = 1
        self.__ngl = ngl

        self.setNumFields(fields)
        numComp = [1]

        dim = self.getDimension()

        numDofVel = [ 1*dim , (ngl-2)*dim , dim * (ngl-2)**2 ]

        bcIs = self.getStratumIS('marker', 1)
        velSec = self.createSection(numComp, numDofVel, 0, bcPoints=[bcIs])
        velSec.setFieldName(0, 'velocity')

        self.setDefaultSection(velSec)
        self.velSec = self.getDefaultGlobalSection()

    def createElement(self):
        assert self.__ngl, "NGL Not defined"
        self.__elem = Spectral2(self.__ngl, self.getDimension())
        self.computeFullCoordinates()

    def getNGL(self):
        return self.__ngl

    def getLocalVelocityDofsFromCell(self, cell):
        points, _ = self.getTransitiveClosure(cell)
        arr = np.zeros(0, dtype=np.int32)
        points = self.reorderEntities(points)
        for poi in points:
            arrtmp = np.arange(*self.getPointLocal(poi))
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

    def getTotalElements(self):
        firstCell, lastCell = self.getHeightStratum(0)
        return lastCell - firstCell

    def getCellRange(self):
        return self.getHeightStratum(0)

    def computeFullCoordinates(self):
        dim = self.getDimension()
        self.fullCoordVec = self.createLocalVec()
        self.fullCoordVec.setName('Coordinates')

        coordsComponents = dim

        spElem = self.__elem

        for cell in range(*self.getHeightStratum(0)):
            coords = self.getCellCornersCoords(cell)
            coords.shape = (2** coordsComponents , coordsComponents)
            indices = self.getLocalVelocityDofsFromCell(cell)
            elTotNodes = spElem.nnode
            totCoord = np.zeros((coordsComponents*elTotNodes))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx]@coords).T

            self.fullCoordVec.setValues(indices.astype(np.int32), totCoord)

        self.fullCoordVec.assemble()

    def getCellCornersCoords(self, cell):
        start = self.getHeightStratum(0)[0]
        coordinates = self.getCoordinatesLocal()
        coordSection = self.getCoordinateSection()
        return self.vecGetClosure(coordSection,
                                         coordinates,
                                         cell+start)

    def getAllNodes(self):
        raise Exception("Not implemented")

    def getNodesCoordinates(self, nodes=None, indices=None, label=None):
        """
        nodes: [Int]
        """
        dim = self.getDimension()
        try:
            assert nodes is not None
            indices = self.indicesManager.mapNodesToIndices(nodes, dim)
            arr = self.fullCoordVec.getValues(indices).reshape((len(nodes),dim))
        except AssertionError:
            assert indices is not None
            numOfNodes = floor(len(indices) / dim)
            arr = self.fullCoordVec.getValues(indices).reshape((numOfNodes,dim))
        return arr
        # raise Exception("Not implemented yet")

    def getEdgesWidth(self):
        startEnt, _ = self.getDepthStratum(1)
        coordinates = self.getCoordinatesLocal()
        coordSection = self.getCoordinateSection()
        coord = self.vecGetClosure(coordSection, coordinates, startEnt).reshape(2,self.dim)
        coord = coord[1] - coord[0]
        norm = np.linalg.norm(coord)
        return norm

    def getDofsRange(self):
       sec = self.indicesManager.getGlobalIndicesSection()
       return sec.getOffsetRange()

    def computeLocalMatrices(self, cellNum):
        cornerCoords = self.getCellCornersCoords(cellNum)
        return self.__elem.getElemKLEMatrices(cornerCoords)

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
