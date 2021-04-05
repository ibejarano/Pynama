import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from .indices import IndicesManager
from .elements.spectral import Spectral
from boundaries.boundary_conditions import BoundaryConditions
from .dmplex import GmshDom, BoxDom
import numpy as np
import logging
from mpi4py import MPI
from math import pi, floor
import copy

class Domain:
    comm = PETSc.COMM_WORLD
    def __init__(self):
        self.logger = logging.getLogger(f"{[self.comm.rank]}:Domain:")
        self.__meshType = None
        self.__bc = None
        self.__ngl = None
        self.__dm = None
        self.__elem = None

    def create(self):
        assert self.domData, "Domain not defined"

        if 'box-mesh' in self.domData:
            self.__meshType = 'box'
            dm = BoxDom()
            cfg = self.domData['box-mesh']
        elif 'gmsh-file' in self.domData:
            self.__meshType = 'gmsh'
            dm = GmshDom()
            cfg = self.domData['gmsh-file']
        else:
            raise Exception("Mesh Type not defined")

        dm.create(cfg)
        self.__dm = dm
        self.__ngl = self.domData['ngl']
        del self.domData
        self.logger.info(f"DMPlex type: {self.__meshType} created")

    def configure(self, inp):
        data = copy.deepcopy(inp)
        if 'domain' in data:
            self.domData = data['domain']

        if 'boundary-conditions' in data:
            self.bcData = data['boundary-conditions']

    def setOptions(self, **kwargs):
        domOpts = ('ngl')
        boxOpts = ('lower', 'upper', 'nelem')
        gmshOpts = ('fileName')
        bcOpts = ('freeSlip', 'noSlip', 'constant')

        for key, val in kwargs.items():
            if key in domOpts:
                self.domData[key] = val
            elif key in boxOpts:
                self.domData['box-mesh'][key] = val
            elif key in gmshOpts:
                self.domData['gmsh'][key] = val
            elif key in bcOpts:
                if key == 'freeSlip':
                    k = 'free-slip'
                elif key == 'noSlip':
                    k='no-slip'
                else:
                    k = key
                self.bcData[k] = val
            else:
                self.logger.warning("Unknown option for domain")

    def setUp(self):
        self.create()
        self.setUpIndexing()
        dim = self.__dm.getDimension()
        self.setUpSpectralElement(Spectral(self.__ngl, dim))
        self.setUpLabels()

        self.setUpBoundaryConditions()
        self.computeFullCoordinates()
        self.setUpBoundaryCoordinates()

    def setUpBoundaryConditions(self):
        assert self.bcData, "Boundary Conditions Not defined"

        bNames = self.__dm.getBordersNames()
        bcs = BoundaryConditions(bNames)
        bcs.setBoundaryConditions(self.bcData)
        boundariesNames = bcs.getNames()
        for bName in boundariesNames:
            nodes = self.__dm.getBorderNodes(bName)
            bcs.setBoundaryNodes(bName, nodes)

        self.__bc = bcs
        del self.bcData

    def setUpBoundaryCoordinates(self):
        bordersWithCoords = self.__bc.getBordersNeedsCoords()
        for borderName in bordersWithCoords:
            inds = self.__bc.getIndicesByName(borderName)
            coords = self.getCoordinates(inds)
            self.__bc.setBoundaryCoords(borderName, coords)
    
    def getDM(self):
        return self.__dm

    def setUpLabels(self):
        self.__dm.setLabelToBorders()

    def getMeshType(self):
        return self.__meshType

    def getBoundaryType(self):
        try:
            return self.__bc.getType()
        except:
            raise Exception("Boundary Type not defined")

    def getDimension(self):
        return self.__dm.getDimension()

    def getDimensions(self):
        dim = self.__dm.getDimension()
        dim_w = 1 if dim == 2 else 3
        dim_s = 3 if dim == 2 else 6
        return dim, dim_w, dim_s

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

    def getFullCoordArray(self):
        dim = self.__dm.getDimension()
        nodes = self.getNumOfNodes()
        return self.__dm.fullCoordVec.getArray().reshape((nodes, dim))

    def getCellCentroid(self, cell):
        dim = self.__dm.getDimension()
        cornerCoords = self.__dm.getCellCornersCoords(cell).reshape((2**dim), dim)
        return np.mean(cornerCoords, axis=0)

    def getNodesCoordinates(self, nodes):
        return self.__dm.getNodesCoordinates(nodes=nodes)

    def getCoordinates(self, indices):
        return self.__dm.getNodesCoordinates(indices=indices)

    def destroyCoordVec(self):
        self.__dm.fullCoordVec.destroy()

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

    def getBoundarySharedIndices(self):
        return self.__bc.getSharedIndices()

    def getNodesOverline(self, line: str, val: float, invert=False):
        assert line in ['x', 'y']
        dim = self.__dm.getDimension()
        dof, orderDof = (0,1) if line == 'x' else (1,0)
        coords = self.__dm.fullCoordVec.getArray()
        nodes = np.where(coords[dof::dim] == val)[0]
        coords = coords[nodes*dim+orderDof]
        tmp = np.stack( (coords, nodes), axis=1)
        tmp = np.sort(tmp.view('i8,i8'), order=['f0'], axis=0).view(np.float)
        coords = tmp[:,0]
        nodes = tmp[:,1].astype(int) 
        return nodes, coords

    def getNodeSeparationIBM(self):
        dm = self.__dm
        ## FIRST GIVE ME ONE EDGE
        eWidth = dm.getEdgesWidth()
        val = eWidth / (self.__ngl - 1)
        return val

    def getFreeStreamVelocity(self):
        vel = self.__bc.getFreeStreamVelocity()
        return np.linalg.norm(vel)

    # -- Mat Index Generator --
    def getMatIndices(self):
        return self.__dm.getMatIndices()

    # -- Indices -- 
    def getGlobalIndicesDirichlet(self):
        fsIndices = self.__bc.getIndicesByType('free-slip')
        return fsIndices
        # return self.__dm.getGlobalIndicesDirichlet()

    def getNodesDirichlet(self, collect=False):
        nodes = self.__bc.getNodesByType('free-slip', allGather=collect)
        return nodes

    def getGlobalIndicesNoSlip(self):
        nsIndices = self.__bc.getIndicesByType('no-slip')
        return nsIndices
        # return self.__dm.getGlobalIndicesNoSlip()

    def getNodesNoSlip(self, collect=False):
        nodes = self.__bc.getNodesByType('no-slip', allGather=collect)
        return nodes

    def getTangDofs(self, collect=False):
        return self.__bc.getNoSlipTangDofs(allGather=collect)

    def getNormalDofs(self, collect=False):
        return self.__bc.getNoSlipNormalDofs(allGather=collect)        

    # -- Mat Building --
    def getNodesRange(self):
        return self.__dm.getNodesRange()

    def getConnectivity(self):
        return self.__dm.getConnectivityNodes()

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

    # -- apply Boundary Conditions to vec 
    def applyBoundaryConditions(self, vec, varName, t=None, nu=None):
        assert self.__bc, "Boundary conditions not defined"
        self.__bc.setValuesToVec(vec, varName, t, nu)

    def applyBoundaryConditionsNS(self, vec, varName, t=None, nu=None):
        assert self.__bc, "Boundary conditions not defined"
        self.__bc.setTangentialValuesToVec(vec, varName, t, nu)

    # -- apply values to vec

    def applyValuesToVec(self, nodes, vals, vec):
        return self.__dm.applyValuesToVec(nodes, vals, vec)

    def applyFunctionVecToVec(self,nodes, f_vec, vec, dof):
        return self.__dm.applyFunctionVecToVec(nodes, f_vec, vec, dof)

    # -- view methods
    def view(self):
        print("Domain info")
        if self.__dm == None:
            print(f"Domain not Setted up")
        print(f"Domain dimensions: {self.__dm.getDimension()}")
        print(f"Mesh Type : {self.__meshType}")
        print(f"Element Type : {self.__elem}")
        print(f"Total number of Elements: {self.getNumOfElements()}")
        print(f"Total number of Nodes: {self.getNumOfNodes()}")
        
        if self.__bc != None:
            print(self.__bc)
        else:
            print("Boundary conditions not defined")

    def viewNodesCoords(self):
        if not self.comm.rank:
            print(" ===== Nodes Coordinates =====")
            print(" Proc Num | Node Global | Node Local |  Coord ")
        coordArr = self.__dm.fullCoordVec.getArray()
        dim = self.__dm.getDimension()
        totalNodes = int(len(coordArr) / dim)
        for node in range(totalNodes):
            print(f"{self.comm.rank:9} | {(node+self.__dm.startNode):11} | {node:10} | {coordArr[node*dim:node*dim+dim]}")