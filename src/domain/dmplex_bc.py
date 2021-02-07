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
        elif self.dim == 3:
            self.namingConvention = ["back", "front", "down", "up", "right", "left"]

        self.markBoundaryFaces('boundary', 0)
        faces = self.getStratumIS('boundary', 0)
        for f in faces.getIndices():
            cell = self.getSupport(f)
            self.setLabelValue('boundary', cell, 1)

    def setFemIndexing(self, ngl, bc=True, fields=['velocity', 'vorticity']):
        self.__ngl = ngl
        self.setNumFields(len(fields))

        dim = self.getDimension()
        dim_w = 1 if dim == 2 else 3
        numComp = [dim, dim_w]
        numDofs = []
        for dofs in numComp:
            numDofPerPoint = [ 1*dofs , dofs * (ngl-2) , dofs * (ngl-2)**2 ]
            numDofs.append(numDofPerPoint)

        if bc:
            bcIs = self.getStratumIS('marker', 1)
            bcIs = [bcIs] * len(fields)
            sec = self.createSection(numComp, numDofs, [0, 1] , bcPoints=bcIs)
        else:
            sec = self.createSection(numComp, numDofs)
        
        for i, fieldName in enumerate(fields):
            sec.setFieldName(i, fieldName)

        self.setDefaultSection(sec)
        self.sec = self.getDefaultGlobalSection()

        names, ids, dms = self.createFieldDecomposition()

        self.velDM, self.vortDM = dms
        self.setUpVecs()

    def setUpVecs(self):
        vort = self.getLocalVorticity()
        vort.setName("vorticity")
        vel = self.getLocalVelocity()
        vel.setName("velocity")
        self.restoreLocalVelocity(vel)
        self.restoreLocalVorticity(vort)
        
    def getNGL(self):
        return self.__ngl

    def computeFullCoordinates(self, spElem):
        dim = self.getDimension()
        fullCoordVec = self.velDM.createLocalVec()
        fullCoordVec.setName('Coordinates')

        coordsComponents = dim
        startCell, _ = self.getHeightStratum(0)

        for cell in range(*self.getHeightStratum(0)):
            coords = getCellCornersCoords(self, startCell ,cell)
            coords.shape = (2** coordsComponents , coordsComponents)
            indices = getLocalDofsFromCell(self.velDM, cell)
            elTotNodes = spElem.nnode
            totCoord = np.zeros((coordsComponents*elTotNodes))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx]@coords).T

            fullCoordVec.setValues(indices.astype(np.int32), totCoord)
        fullCoordVec.assemble()
        return fullCoordVec

    def getGlobalVelocity(self):
        return self.velDM.getGlobalVec()

    def getLocalVelocity(self):
        return self.velDM.getLocalVec()

    def getGlobalVorticity(self):
        return self.vortDM.getGlobalVec()

    def getLocalVorticity(self):
        return self.vortDM.getLocalVec()

    def restoreGlobalVelocity(self, vec):
        self.velDM.restoreGlobalVec(vec)

    def restoreLocalVelocity(self, vec):
        self.velDM.restoreLocalVec(vec)
    
    def restoreLocalVorticity(self, vec):
        self.vortDM.restoreLocalVec(vec)

    def velocityLocalToGlobal(self, glVec, locVec):
        self.velDM.globalToLocal(glVec, locVec)

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