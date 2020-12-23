import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from .indices import IndicesManager
from .elements.spectral import Spectral
import numpy as np
import logging
from mpi4py import MPI
from math import pi, floor

def checkOption(optName, optKwargs):
    OptDB = PETSc.Options()
    optStr = OptDB.getString(optName, False)
    if optStr:
        print(f"Setting {optName} from options: {optStr}")
        return optStr
    elif optName in optKwargs:
        print(f"Setting {optName} from args: {optKwargs[optName]}")
        return optKwargs[optName]
    else:
        return False

class Domain:
    def __init__(self, data:dict={}, comm=PETSc.COMM_WORLD, **kwargs):
        # 1. Gmsh file wins to all
        self.logger = logging.getLogger(f"{[comm.rank]} - Domain:")
        domainData = checkOption("gmsh-file", kwargs)
        if domainData:
            self.logger.info("Setting up with options: gmsh-file")
            self.__meshType = 'gmsh'
        elif 'gmsh-file' in data:
            self.__meshType = 'gmsh'
            domainData = data['gmsh-file']
            self.logger.info("Setting up with gmsh file from yaml")
        else:
            self.__meshType = 'box'
            if 'box-mesh' in data:
                domainData = data['box-mesh']
                self.logger.info("Setting up with Box from yaml")
                availableOptions = ("nelem", "lower", "upper")
                for opt in availableOptions:
                    optConfigured = checkOption(opt, kwargs)
                    if optConfigured:
                        self.logger.info(f"Setting {opt} with options")
                        try:
                            optConfigured = eval(optConfigured)
                        except:
                            optConfigured = optConfigured
                        domainData[opt] = optConfigured

            else:
                raise Exception("Domain not defined")

        nglOpt = checkOption("ngl", kwargs)
        if nglOpt:
            self.__ngl = nglOpt
            self.logger.info(f"Setting NGL w/ options : {self.__ngl}")
        else:
            self.__ngl = data['ngl']
            self.logger.info(f"NGL : {self.__ngl}")

        self.createDomain(domainData)
        self.setUpIndexing()
        dim = self.__dm.getDimension()
        self.setUpSpectralElement(Spectral(self.__ngl, dim))

    def setUp(self):
        self.setUpLabels()
        self.setUpBoundaryConditions()

    def setUpBoundaryConditions(self, fsFaces =[], nsFaces=[]):
        self.__fsFaces = fsFaces
        self.__nsFaces = nsFaces
        self.__dm.setBoundaryCondition(self.__fsFaces, self.__nsFaces)

    def setUpLabels(self):
        self.__dm.setLabelToBorders()

    def createDomain(self, inp):
        if self.__meshType == 'box':
            dm = BoxDom(inp)
        elif self.__meshType == 'gmsh':
            dm = GmshDom(inp)
        else:
            raise Exception("Mesh Type not defined")

        self.__dm = dm

    def getMeshType(self):
        return self.__meshType

    def getDimension(self):
        return self.__dm.getDimension()

    def getNGL(self):
        return self.__ngl

    def getNumOfElements(self):
        return self.__dm.getTotalElements()

    def setUpIndexing(self):
        self.__dm.setFemIndexing(self.__ngl)

    def setUpSpectralElement(self, elem):
        self.__elem = elem

    # -- Coordinates methods ---
    def getExtremeCoords(self):
        lower, upper = self.__dm.getBoundingBox()
        return lower, upper

    def computeFullCoordinates(self):
        self.__dm.computeFullCoordinates(self.__elem)

    def getFullCoordVec(self):
        return self.__dm.fullCoordVec

    def getCellCentroid(self, cell):
        dim = self.__dm.getDimension()
        cornerCoords = self.__dm.getCellCornersCoords(cell).reshape((2**dim), dim)
        return np.mean(cornerCoords, axis=0)

    def getNodesCoordinates(self, nodes):
        return self.__dm.getNodesCoordinates(nodes=nodes)

    # -- Get / SET Nodes methods --
    def getNumOfNodes(self):
        return self.__dm.getTotalNodes()

    def getBoundaryNodes(self):
        return self.__dm.getNodesFromLabel("External Boundary")

    def getAllNodes(self):
        return self.__dm.getAllNodes()

    def getNodesCoordsFromEntities(self, entities):
        nodes = self.__dm.getGlobalNodesFromEntities(entities, shared=True)
        coords = self.__dm.getNodesCoordinates(nodes)
        return nodes, coords

    def getBorderNodesWithNormal(self, cell, cellNodes):
        return self.__dm.getBorderNodesWithNormal(cell, cellNodes)

    def getBorderNodes(self, borderName):
        return self.__dm.getBorderNodes(borderName)

    # -- Mat Index Generator --
    def getMatIndices(self):
        return self.__dm.getMatIndices()

    # -- Indices -- 
    def getGlobalIndicesDirichlet(self):
        return self.__dm.getGlobalIndicesDirichlet()

    def getGlobalIndicesNoSlip(self):
        return self.__dm.getGlobalIndicesNoSlip()

    # -- Mat Building --
    def getLocalCellRange(self):
        return self.__dm.cellStart, self.__dm.cellEnd

    def computeLocalKLEMats(self, cell):
        cornerCoords = self.__dm.getCellCornersCoords(cell)
        localMats = self.__elem.getElemKLEMatrices(cornerCoords)
        nodes = self.__dm.getGlobalNodesFromCell(cell, shared=True)
        # Build velocity and vorticity DoF indices
        indicesVel = self.__dm.getVelocityIndex(nodes)
        indicesW = self.__dm.getVorticityIndex(nodes)
        inds = (indicesVel, indicesW)
        return nodes, inds , localMats

    def computeLocalOperators(self, cell):
        cornerCoords = self.__dm.getCellCornersCoords(cell)
        localOperators = self.__elem.getElemKLEOperators(cornerCoords)
        nodes = self.__dm.getGlobalNodesFromCell(cell, shared=True)
        return nodes, localOperators

    # -- apply values to vec

    def applyValuesToVec(self, nodes, vals, vec):
        return self.__dm.applyValuesToVec(nodes, vals, vec)

    def applyFunctionVecToVec(self,nodes, f_vec, vec, dof):
        return self.__dm.applyFunctionVecToVec(nodes, f_vec, vec, dof)


    def view(self):
        print("Domain info")
        if self.__dm == None:
            print(f"Domain not Setted up")
        print(f"Domain dimensions: {self.__dm.getDimension()}")
        print(f"Mesh Type : {self.__meshType}")
        print(f"Element Type : {self.__elem}")
        print(f"Total number of Elements: {self.getNumOfElements()}")
        print(f"Total number of Nodes: {self.getNumOfNodes()}")

class DMPlexDom(PETSc.DMPlex):
    comm = MPI.COMM_WORLD
    def __init__(self):
        self.logger = logging.getLogger(f"[{self.comm.rank}] Class")
        self.logger.debug("Domain Instance Created")
        self.createLabel('External Boundary')
        self.markBoundaryFaces('External Boundary',0)
        self.distribute()
        self.dim = self.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

        if not self.comm.rank:
            self.logger.debug("DM Plex Box Mesh created")

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
        self.logger.debug("Full coord vec size %s", self.fullCoordVec.size)

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

    def getNodesOverline(self, line: str, val: float, invert=False):
        assert line in ['x', 'y']
        dof, orderDof = (0,1) if line == 'x' else (1,0)
        coords = self.fullCoordVec.getArray()
        nodes = np.where(coords[dof::self.dim] == val)[0]
        coords = coords[nodes*self.dim+orderDof]
        tmp = np.stack( (coords, nodes), axis=1)
        tmp = np.sort(tmp.view('i8,i8'), order=['f0'], axis=0).view(np.float)
        coords = tmp[:,0]
        nodes = tmp[:,1].astype(int) 
        return nodes, coords

    def getVecArrayFromNodes(self, vec, nodes):
        vecArr = vec.getArray()
        arr_x = vecArr[nodes*self.dim]
        arr_y = vecArr[nodes*self.dim+1]
        return arr_x, arr_y

class BoxDom(DMPlexDom):
    """Estrucuted DMPlex Mesh"""
    def __init__(self, data):
        lower = data['lower']
        upper = data['upper']
        faces = data['nelem']
        self.createBoxMesh(faces=faces, lower=lower, upper=upper, simplex=False)
        super().__init__()

class GmshDom(DMPlexDom):
    """Unstructured DMPlex Mesh"""
    def __init__(self, fileName: str):
        self.createFromFile(fileName)
        super().__init__()

if __name__ == "__main__":
    data = {"ngl":3, "box_mesh": {
        "nelem": [2,2],
        "lower": [0,0],
        "upper": [1,1]
    }}

    domain = Domain(data)
