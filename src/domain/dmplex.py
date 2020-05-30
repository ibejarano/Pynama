from common.baseclass import Base
from petsc4py import PETSc
from domain.indices import IndicesManager
import numpy as np

class DMPlexDom(Base):
    def __init__(self, dim):
        """Aca dijimos que una clase madre le va a pasar esta data"""
        super().__init__(dim)
        self.dm = PETSc.DMPlex()
        self.logger.debug("Domain Instance Created")

    def setUpDmPlex(self, lower, upper, faces):
        self.dm.createBoxMesh(faces=faces, lower=lower, upper=upper, simplex=False, comm=self.comm)

        self.dm.createLabel('marco')
        self.dm.markBoundaryFaces('marco',0)
        self.dm.distribute()

        if not self.comm.rank:
            self.logger.debug("DM Plex Box Mesh created")

    def setFemIndexing(self, ngl):
        # self._ngl = ngl
        fields = 1
        componentsPerField = 1
        # cosas DM
        self.dm.setNumFields(fields)

        self.indicesManager = IndicesManager(self.dim, ngl ,self.comm)
        numComp, numDof = self.indicesManager.getNumCompAndNumDof(componentsPerField, fields)
        
        indSec = self.dm.createSection(numComp, numDof)
        indSec.setFieldName(0, 'FEM/SEM indexing')
        indSec.setUp()
        self.indicesManager.setLocalIndicesSection(indSec)

        self.dm.setDefaultSection(indSec)
        indGlobSec = self.dm.getDefaultGlobalSection()
        self.cellStart, self.cellEnd = self.dm.getHeightStratum(0)
        self.indicesManager.setGlobalIndicesSection(indGlobSec)

        if not self.comm.rank:
            self.logger.debug("FEM/SEM Indexing SetUp")


    def computeFullCoordinates(self, spElem):
        # self.logger = logging.getLogger("[{}] DomainMin Compute Coordinates".format(self.comm.rank))
        coordsComponents = self.dim
        numComp, numDof = self.indicesManager.getNumCompAndNumDof(coordsComponents, 1)
        fullCoordSec = self.dm.createSection(numComp, numDof)
        fullCoordSec.setFieldName(0, 'Vertexes')
        fullCoordSec.setUp()
        self.dm.setDefaultSection(fullCoordSec)
        self.fullCoordVec = self.dm.createGlobalVec()
        self.fullCoordVec.setName('NodeCoordinates')
        self.logger.debug("Full coord vec size %s", self.fullCoordVec.size)

        for cell in range(self.cellEnd - self.cellStart):
            coords = self.getCellCornersCoords(cell)
            elTotNodes = spElem.nnode
            coords.shape = (2**self.dim, coordsComponents)
            # self.logger.debug('coordenadas %s',coords)
            # nodosGlobales = self.getElemNodes(elem, "global")[0]
            cellEntities, orientations = self.dm.getTransitiveClosure(cell)
            nodosGlobales = self.indicesManager.mapEntitiesToNodes(cellEntities, orientations)
            # self.logger.debug("Nodos Globales %s", nodosGlobales)
            
            # indicesGlobales = self.getElemIndices(elem, 'global', coordsComponents)
            indicesGlobales = self.indicesManager.mapNodesToIndices(nodosGlobales, coordsComponents)
            totCoord = np.mat(np.zeros((coordsComponents*elTotNodes, 1)))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx] * coords).T
            
            # self.logger.debug("total coords %s", totCoord)
            # self.logger.debug("indices %s", indicesGlobales)
            self.fullCoordVec.setValues(indicesGlobales, totCoord)

        self.fullCoordVec.assemble()
        self.nodes = [int(node/coordsComponents) for node in range(self.fullCoordVec.owner_range[0],
        self.fullCoordVec.owner_range[1], coordsComponents)]

    def getCellCornersCoords(self, cell):
        coordinates = self.dm.getCoordinatesLocal()
        coordSection = self.dm.getCoordinateSection()
        if cell + self.cellStart >= self.cellEnd:
            raise Exception('elem parameter must be in local numbering!')
        return self.dm.vecGetClosure(coordSection,
                                         coordinates,
                                         cell+self.cellStart)

    def setLabelToBorders(self):

        label = 'cfgfileBC'
        self.dm.createLabel(label)
        for faceNum in self.dm.getLabelIdIS("Face Sets").getIndices():
            Faces= self.dm.getStratumIS("Face Sets", faceNum).getIndices()
            borderNum = faceNum - 1
            for Face in Faces: 
                entitiesToLabel=self.dm.getTransitiveClosure(Face)[0]
                for entity in entitiesToLabel: 
                    oldVal = self.dm.getLabelValue(label, entity)
                    if oldVal >= 0:
                        self.dm.clearLabelValue(label, entity, oldVal)
                        self.dm.setLabelValue(label, entity,
                                            2**borderNum | oldVal)
                    else:
                        self.dm.setLabelValue(label, entity,
                                            2**borderNum)
        if not self.comm.rank:
            self.logger.debug("Labels creados en borders")

    def getDMConectivityMat(self):
        localIndicesSection = self.indicesManager.getLocalIndicesSection()
        self.dm.setDefaultSection(localIndicesSection)
        return self.dm.createMat()

    def getGlobalIndicesDirichlet(self):
        indicesDIR = self.indicesManager.getDirichletIndices()
        return indicesDIR


    def readBoundaryCondition(self, BCdict):
        tag2BCdict = dict()
        BCset = set()
        # Existing tag values in 'cfgfileBC' label
        BClabelVals = self.dm.getLabelIdIS('cfgfileBC').getIndices()
        for val in BClabelVals:
            labelVal = val
            tag2BCdict[val] = list()
            bcNum = 0
            while labelVal:
                if labelVal & 1:
                    secName = "bc-%02d" % bcNum
                    # print(secName)
                    tag2BCdict[val].append(secName)
                    BCset.add(secName)
                labelVal = labelVal >> 1  # Shift 1 bit to delete first bit
                bcNum += 1

        # Keep only those entries in BCdict that are necessary
        BCdict = {k: v for k, v in BCdict.items() if k in BCset}

        BC2nodedict = dict()
        for bc in BCset:
            BC2nodedict[bc] = set()
        node2tagdict = dict()
        for tag in tag2BCdict:
            # DMPlex points with val tag value
            tagDMpoints = self.dm.getStratumIS('cfgfileBC', tag).\
                            getIndices()

            for poi in tagDMpoints:
                # indPoi, ownPoi = self.getInd(poi, "global")
                nodes, ownNodes = self.indicesManager.getGlobalNodes(poi)
                if ownNodes:
                    for node in nodes:
                        # FIXME: node coordinates 2D or 3D
                        ind = [node*self.dim + d for d in range(self.dim)]
                        xyz = self.fullCoordVec.getValues(ind)
                        node2tagdict[node] = [tag, xyz]

                        for bc in tag2BCdict[tag]:
                            BC2nodedict[bc].add(node)

        self.indicesManager.setDirichletIndices(BC2nodedict)
        self._bcConditions = BCdict
        return tag2BCdict, node2tagdict


    def getGlobalNodesFromCell(self, cell, shared):
        entities, orientations = self.dm.getTransitiveClosure(cell)
        nodes = self.indicesManager.mapEntitiesToNodes(entities, orientations, shared)
        return nodes

    def getVelocityIndex(self, nodes):
        indices = self.indicesManager.mapNodesToIndices(nodes, self.dim)
        return indices

    def getVorticityIndex(self, nodes):
        indices = self.indicesManager.mapNodesToIndices(nodes, self.dim_w)
        return indices

    def getSrtIndex(self, nodes):
        indices = self.indicesManager.mapNodesToIndices(nodes, self.dim_s)
        return indices
