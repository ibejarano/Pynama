from petsc4py import PETSc
from domain.indices import IndicesManager
import numpy as np
import logging
from mpi4py import MPI

class DMPlexDom(PETSc.DMPlex):
    def __init__(self, lower, upper, faces):
        comm = MPI.COMM_WORLD
        self.createBoxMesh(faces=faces, lower=lower, upper=upper, simplex=False, comm=comm)
        self.logger = logging.getLogger(f"[{self.comm.rank}] Class")
        self.logger.debug("Domain Instance Created")
        self.createLabel('marco')
        self.markBoundaryFaces('marco',0)
        self.distribute()

        self.dim = self.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

        if not self.comm.rank:
            self.logger.debug("DM Plex Box Mesh created")

    def setFemIndexing(self, ngl):
        fields = 1
        componentsPerField = 1
        self.setNumFields(fields)
        dim = self.getDimension()
        self.indicesManager = IndicesManager(dim, ngl ,self.comm)
        numComp, numDof = self.indicesManager.getNumCompAndNumDof(componentsPerField, fields)
        
        indSec = self.createSection(numComp, numDof)
        indSec.setFieldName(0, 'FEM/SEM indexing')
        indSec.setUp()
        self.indicesManager.setLocalIndicesSection(indSec)

        self.setDefaultSection(indSec)
        indGlobSec = self.getDefaultGlobalSection()
        self.cellStart, self.cellEnd = self.getHeightStratum(0)
        self.indicesManager.setGlobalIndicesSection(indGlobSec)

        if not self.comm.rank:
            self.logger.debug("FEM/SEM Indexing SetUp")


    def computeFullCoordinates(self, spElem):
        # self.logger = logging.getLogger("[{}] DomainMin Compute Coordinates".format(self.comm.rank))
        coordsComponents = self.getDimension()
        numComp, numDof = self.indicesManager.getNumCompAndNumDof(coordsComponents, 1)
        fullCoordSec = self.createSection(numComp, numDof)
        fullCoordSec.setFieldName(0, 'Vertexes')
        fullCoordSec.setUp()
        self.setDefaultSection(fullCoordSec)
        self.fullCoordVec = self.createGlobalVec()
        self.fullCoordVec.setName('NodeCoordinates')
        self.logger.debug("Full coord vec size %s", self.fullCoordVec.size)

        for cell in range(self.cellEnd - self.cellStart):
            coords = self.getCellCornersCoords(cell)
            coords.shape = (2** coordsComponents , coordsComponents)
            cellEntities, orientations = self.getTransitiveClosure(cell)
            nodosGlobales = self.indicesManager.mapEntitiesToNodes(cellEntities, orientations)
            indicesGlobales = self.indicesManager.mapNodesToIndices(nodosGlobales, coordsComponents)

            elTotNodes = spElem.nnode
            totCoord = np.mat(np.zeros((coordsComponents*elTotNodes, 1)))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx] * coords).T
            self.fullCoordVec.setValues(indicesGlobales, totCoord)

        self.fullCoordVec.assemble()
        self.nodes = [int(node/coordsComponents) for node in range(self.fullCoordVec.owner_range[0],
        self.fullCoordVec.owner_range[1], coordsComponents)]

    def getCellCornersCoords(self, cell):
        coordinates = self.getCoordinatesLocal()
        coordSection = self.getCoordinateSection()
        if cell + self.cellStart >= self.cellEnd:
            raise Exception('elem parameter must be in local numbering!')
        return self.vecGetClosure(coordSection,
                                         coordinates,
                                         cell+self.cellStart)

    def setLabelToBorders(self):
        label = 'cfgfileBC'
        self.createLabel(label)
        for faceNum in self.getLabelIdIS("Face Sets").getIndices():
            Faces= self.getStratumIS("Face Sets", faceNum).getIndices()
            borderNum = faceNum - 1
            for Face in Faces: 
                entitiesToLabel=self.getTransitiveClosure(Face)[0]
                for entity in entitiesToLabel: 
                    oldVal = self.getLabelValue(label, entity)
                    if oldVal >= 0:
                        self.clearLabelValue(label, entity, oldVal)
                        self.setLabelValue(label, entity,
                                            2**borderNum | oldVal)
                    else:
                        self.setLabelValue(label, entity,
                                            2**borderNum)
        if not self.comm.rank:
            self.logger.debug("Labels creados en borders")

    def getDMConectivityMat(self):
        localIndicesSection = self.indicesManager.getLocalIndicesSection()
        self.setDefaultSection(localIndicesSection)
        return self.createMat()

    def getGlobalIndicesDirichlet(self):
        indicesDIR = self.indicesManager.getDirichletIndices()
        return indicesDIR

    def setBoundaryCondition(self):
        dim = self.getDimension()
        tag2BCdict = dict()
        BCset = set()
        # Existing tag values in 'cfgfileBC' label
        BClabelVals = self.getLabelIdIS('cfgfileBC').getIndices()
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

        BC2nodedict = dict()
        for bc in BCset:
            BC2nodedict[bc] = set()
        node2tagdict = dict()
        for tag in tag2BCdict:
            # DMPlex points with val tag value
            tagDMpoints = self.getStratumIS('cfgfileBC', tag).\
                            getIndices()

            for poi in tagDMpoints:
                # indPoi, ownPoi = self.getInd(poi, "global")
                nodes, ownNodes = self.indicesManager.getGlobalNodes(poi)
                if ownNodes:
                    for node in nodes:
                        # FIXME: node coordinates 2D or 3D
                        ind = [node*dim + d for d in range(dim)]
                        xyz = self.fullCoordVec.getValues(ind)
                        node2tagdict[node] = [tag, xyz]

                        for bc in tag2BCdict[tag]:
                            BC2nodedict[bc].add(node)

        self.indicesManager.setDirichletIndices(BC2nodedict)
        return tag2BCdict, node2tagdict


    def getGlobalNodesFromCell(self, cell, shared):
        entities, orientations = self.getTransitiveClosure(cell)
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

    def getAllNodes(self):
        # TODO: Needs to be tested in parallel
        start, end = self.getChart()
        globalNodes = list()
        for entity in range(start, end):
            globalNodes.extend(self.indicesManager.getGlobalNodes(entity)[0])
        return globalNodes

    def getNodesCoordinates(self, nodes):
        """
        nodes: [Int]
        """
        dim = self.getDimension()
        indices = self.indicesManager.mapNodesToIndices(nodes, dim)
        arr = self.fullCoordVec.getValues(indices).reshape((len(nodes),dim))
        return arr

    def applyFunctionVecToVec(self, nodes, f_vec, vec, dof):
        """
        f_vec: function: returns a tuple with len = dim
        summary; this method needs to map nodes to indices
        """
        coords = self.getNodesCoordinates(nodes)
        for i,coord in enumerate(coords):
            values = f_vec(coord)
            indices = [nodes[i]*dof + pos for pos in range(dof)]
            vec.setValues(indices, values, addv=None)
        return vec

    def applyFunctionScalarToVec(self, nodes, f_scalar, vec):
        """
        f_scalar: function: returns an unique value
        summary: this nodes = indices
        """
        coords = self.getNodesCoordinates(nodes)
        for i,coord in enumerate(coords):
            value = f_scalar(coord)
            vec.setValues(nodes[i], value, addv=None)
        return vec

    def applyValuesToVec(self, nodes, values, vec):
        # with nodes -> indices
        # TODO in applyFunctionToVec it requests very coordenate
        # the function are coords independent in this case.
        dof = len(values)
        assert dof <= self.dim # Not implemented for single value
        if dof == 1: #apply a scalar for every node in vec
            vec.set(values[0])
        else:
            valuesToSet = np.array(values * len(nodes))
            indices = self.getVelocityIndex(nodes)
            vec.setValues(indices, valuesToSet, addv=False)     
        return vec


class DomainElementInterface(object):
    
    def __init__(self, element):
        self.elem = element

    def getTotalNodes(self):   
        return self.elem.nnode