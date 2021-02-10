import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from .indices import IndicesManager
import numpy as np
import logging
from mpi4py import MPI
from math import pi, floor

class DMPlexDom(PETSc.DMPlex):
    comm = PETSc.COMM_WORLD
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"[{self.comm.rank}] Class")
        self.logger.info("DMPLEX Instance Created")

    def create(self):
        self.createLabel('External Boundary')
        self.markBoundaryFaces('External Boundary',0)
        self.distribute()
        self.dim = self.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

        if self.dim == 2:
            self.namingConvention = ["down", "right" , "up", "left"]
        elif self.dim == 3:
            self.namingConvention = ["back", "front", "down", "up", "right", "left"]

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

    def getNGL(self):
        return self.indicesManager.getNGL()

    def getTotalElements(self):
        firstCell, lastCell = self.getHeightStratum(0)
        return lastCell - firstCell

    def getTotalNodes(self):
        totalNodes = self.indicesManager.getTotalNodes()
        return totalNodes

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

        for cell in range(self.cellEnd - self.cellStart):
            coords = self.getCellCornersCoords(cell)
            coords.shape = (2** coordsComponents , coordsComponents)
            cellEntities, orientations = self.getTransitiveClosure(cell)
            nodosGlobales = self.indicesManager.mapEntitiesToNodes(cellEntities, orientations)
            indicesGlobales = self.indicesManager.mapNodesToIndices(nodosGlobales, coordsComponents)

            elTotNodes = spElem.nnode
            totCoord = np.zeros((coordsComponents*elTotNodes))

            for idx, gp in enumerate(spElem.gpsOp):
                totCoord[[coordsComponents * idx + d for d in range(coordsComponents)]] = \
                    (spElem.HCooOp[idx]@coords).T
            self.fullCoordVec.setValues(indicesGlobales, totCoord)

        self.fullCoordVec.assemble()
        self.startNode = int(self.fullCoordVec.owner_range[0]/self.dim)
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

    def getFaceCoords(self, face):
        coordinates = self.getCoordinatesLocal()
        coordSection = self.getCoordinateSection()
        return self.vecGetClosure(coordSection,
                                         coordinates,
                                         face)

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

    def __getBorderEntities(self, name):
        faceNum = self.__mapFaceNameToNum(name)
        try:
            faces = self.getStratumIS("Face Sets", faceNum).getIndices()
        except:
            faces = []
        return faces

    def getNodesFromLabel(self, label, shared=False) -> set:
        nodes = set()
        try:
            entities = self.getStratumIS(label, 0).getIndices()
            # for entity in entities:
            nodes |= self.getGlobalNodesFromEntities(entities,shared=shared)
        except:
            self.logger.warning(f"Label >> {label} << found")
        return nodes

    def getBordersNames(self):
        return self.namingConvention

    def getBordersNodes(self) -> set:
        nodes = set()
        for faceName in self.namingConvention:
            nodes |= set(self.getBorderNodes(faceName))
        return nodes

    def getBorderNodes(self, name):
        entities = self.__getBorderEntities(name)
        nodesSet = set()
        for entity in entities:
            nodes = self.getGlobalNodesFromCell(entity, False)
            nodesSet |= set(nodes)
        return list(nodesSet)

    def __mapFaceNameToNum(self, name):
        """This ordering corresponds to nproc = 1"""
        num = self.namingConvention.index(name) + 1
        return num

    def getGlobalIndicesDirichlet(self):
        indicesDIR = self.indicesManager.getDirichletNodes()
        return indicesDIR

    def getGlobalIndicesNoSlip(self):
        indicesNS = self.indicesManager.getNoSlipNodes()
        return indicesNS

    def setBoundaryCondition(self, freeSlipFaces = [], noSlipFaces = []):
        # 1. pasarle parametros no slip y free slip
        # el parametro tiene que ser el nombre de la cara correspondiente
        # 2. agregar setNSIndices() al indicesManager
        # allBorderNodes = self.getBordersNodes()
        if len(freeSlipFaces) or len(noSlipFaces):
            for fsFace in freeSlipFaces:
                faceNodes = self.getBorderNodes(fsFace)
                self.indicesManager.setDirichletNodes(set(faceNodes))
            for nsFace in noSlipFaces:
                faceNodes = self.getBorderNodes(nsFace)
                self.indicesManager.setNoSlipNodes(set(faceNodes))
        else:
            allBorderNodes = self.getBordersNodes()
            self.indicesManager.setDirichletNodes(allBorderNodes)

    def getGlobalNodesFromCell(self, cell, shared):
        entities, orientations = self.getTransitiveClosure(cell)
        nodes = self.indicesManager.mapEntitiesToNodes(entities, orientations, shared)
        return nodes

    def getGlobalNodesFromEntities(self, entities, shared):
        nodes = set()
        for entity in entities:
            entities, orientations = self.getTransitiveClosure(entity)
            current = self.indicesManager.mapEntitiesToNodes(entities, orientations, shared)
            nodes |= set(current)
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
            globalNodes.extend(self.indicesManager.getGlobalNodes(entity, shared=False)[0])
        return globalNodes

    def getNodesCoordinates(self, nodes=None, indices=None):
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

    def getEdgesWidth(self):
        # bounding for dim = 2 are edges and for dim = 3 are faces
        startEnt, endEnt = self.getDepthStratum(1)
        coordinates = self.getCoordinatesLocal()
        coordSection = self.getCoordinateSection()

        # TODO: For future implementations:
        # for e in range(startEnt, endEnt):
        #     coord = self.vecGetClosure(coordSection,
        #                                  coordinates,
        #                                  e)
        coord = self.vecGetClosure(coordSection, coordinates, startEnt).reshape(2,self.dim)
        coord = coord[1] - coord[0]
        norm = np.linalg.norm(coord)
        return norm


    def getBorderNodesWithNormal(self, cell, intersect):
        nodes = list()
        normals = list()
        localEntities = set(self.getTransitiveClosure(cell)[0])
        for faceName in self.namingConvention:
            globalEntitiesBorders = set(self.__getBorderEntities(faceName))
            localEntitiesBorders = globalEntitiesBorders & localEntities 
            if localEntitiesBorders:
                localEntitiesBorders = list(localEntitiesBorders)
                borderNodes = self.getGlobalNodesFromEntities(localEntitiesBorders, shared=True)
                # Si el conjunto borderNodes de la cara no esta completamente contenido en intersect, entonces pertenece a otro tipo de Boundary cond.
                if not (set(borderNodes) - intersect):
                    normal = self.computeCellGeometryFVM(localEntitiesBorders[0])[2]
                    indexNormal = list(np.abs(normal)).index(1)
                    normals.append(indexNormal)
                    nodes.append(borderNodes)
        return nodes, normals

    def applyFunctionVecToVec(self, nodes, f_vec, vec, dof):
        """
        f_vec: function: returns a tuple with len = dim
        summary; this method needs to map nodes to indices
        """
        coords = self.getNodesCoordinates(nodes)
        inds = [node*dof + pos for node in nodes for pos in range(dof)]
        values = np.array(list(map(f_vec, coords)))
        vec.setValues(inds, values, addv=False)
        return vec

    def applyFunctionScalarToVec(self, nodes, f_scalar, vec):
        """
        f_scalar: function: returns an unique value
        summary: this nodes = indices
        """
        coords = self.getNodesCoordinates(nodes)
        values = np.array(list(map(f_scalar, coords)))
        vec.setValues(nodes, values, addv=False)
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

    def getDMConectivityMat(self):
        localIndicesSection = self.indicesManager.getLocalIndicesSection()
        self.setDefaultSection(localIndicesSection)
        return self.createMat()

    def getNodesRange(self):
       sec = self.indicesManager.getGlobalIndicesSection()
       return sec.getOffsetRange()

    ## Matrix build ind
    # @profile
    def getMatIndices(self):
        conecMat = self.getDMConectivityMat()
        rStart, rEnd = conecMat.getOwnershipRange()
        locElRow = rEnd - rStart
        ind_d = np.zeros(locElRow, dtype=set)
        alt_d = np.zeros(locElRow, dtype=np.int32)
        ind_o = np.zeros(locElRow, dtype=set)
        alt_o = np.zeros(locElRow, dtype=np.int32)

        for row in range(rStart, rEnd):
            cols, _ = conecMat.getRow(row)
            locRow = row - rStart
            mask_diag = np.logical_and(cols >= rStart,cols < rEnd)
            mask_off = np.logical_or(cols < rStart,cols >= rEnd)
            ind_d[locRow] = set(cols[mask_diag])
            alt_d[locRow] = len(ind_d[locRow]) 
            ind_o[locRow] = set(cols[mask_off])
            alt_o[locRow] = len(ind_o[locRow]) 
        conecMat.destroy()
        # d_nnz_ind = [len(indSet) for indSet in ind_d]
        # o_nnz_ind = [len(indSet) for indSet in ind_o]

        # TODO : Fix the line below for parallel
        # TODO : this line is not doing anything at all
        # d_nnz_ind = [x if x <= locElRow else locElRow for x in d_nnz_ind]
        # self.logger.info(f"d_nnz_ind_old {o_nnz_ind}")
        # self.logger.info(f"new one {alt_o}  ")
        # exit()
        return rStart, rEnd, alt_d, alt_o, ind_d, ind_o

    def getConnectivityNodes(self):
        conecMat = self.getDMConectivityMat()
        rStart, rEnd = conecMat.getOwnershipRange()
        locElRow = rEnd - rStart
        ind_d = np.zeros(locElRow, dtype=set)
        ind_o = np.zeros(locElRow, dtype=set)

        nnz_diag = np.zeros(locElRow, dtype=np.int32)
        nnz_off = np.zeros(locElRow, dtype=np.int32)

        for row in range(rStart, rEnd):
            cols, _ = conecMat.getRow(row)
            locRow = row - rStart
            mask_diag = np.logical_and(cols >= rStart,cols < rEnd)
            mask_off = np.logical_or(cols < rStart,cols >= rEnd)
            ind_d[locRow] = set(cols[mask_diag])
            nnz_diag[locRow] = len(ind_d[locRow])
            ind_o[locRow] = set(cols[mask_off])
            nnz_off[locRow] = len(ind_o[locRow]) 
        conecMat.destroy()
        return ind_d, ind_o, nnz_diag, nnz_off

    def getVecArrayFromNodes(self, vec, nodes):
        vecArr = vec.getArray()
        arr_x = vecArr[nodes*self.dim]
        arr_y = vecArr[nodes*self.dim+1]
        return arr_x, arr_y

class BoxDom(DMPlexDom):
    """Estrucuted DMPlex Mesh"""
    def create(self, data):
        lower = data['lower']
        upper = data['upper']
        faces = data['nelem']
        self.createBoxMesh(faces=faces, lower=lower, upper=upper, simplex=False)
        self.logger.info("Box mesh generated")
        super().create()

class GmshDom(DMPlexDom):
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
