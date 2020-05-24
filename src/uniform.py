import sys
import petsc4py

petsc4py.init(sys.argv)

from cases.base_problem import BaseProblem
from matrices.mat_generator import Mat
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc

class UniformFlow(BaseProblem):
    def __init__(self, comm):
        super().__init__(comm)

    def setUpBoundaryConditions(self, inputData):
        #aca 
        self.dom.setLabelToBorders()
        self.tag2BCdict, self.node2tagdict = self.dom.readBoundaryCondition(inputData)

    def setBorderValues(self, borderValues):
        """
        borderValues: [ float ] len 2 or 3, represents
        [ velx , vely ( ,velz)]
        This function stores values in the specific
        d.o.f. of a velocity global PETSc Vec
        """
        raise NotImplementedError

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        fakeConectMat = self.dom.getDMConectivityMat()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()
        self.mat.createEmptyKLEMats(fakeConectMat, globalIndicesDIR)

    def buildKLEMats(self):
        # boundaryConditions = self.dom.bcConditions
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        indices2onefs = set()  # idem for FS solution
        boundaryNodes = set(self.node2tagdict.keys())

        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            self.logger.debug("DMPlex cell: %s", cell)

            # indices, plexPoi = self.getElemNodes(cell,"global", shared = False)
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=False)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
            # indicesSrt = self.dom.getSrtIndex(nodes)
            
            nodeBCintersect = boundaryNodes & set(nodes)
            # self.logger.debug("te intersecto: %s", nodeBCintersect)
            
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions

            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                # FIXME : No importa el bc, #TODO cuando agregemos NS si importa
                for dof in range(self.dim):
                    dofSetFSNS.add(localBoundaryNode*self.dim + dof)

            dofFree = list(set(range(len(indicesVel)))
                           - dofFreeFSSetNS - dofSetFSNS)
            dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
            dofFreeFSSetNS = list(dofFreeFSSetNS)
            dofSetFSNS = list(dofSetFSNS)
            # global counterparts of dof sets
            # gldofFreeFSSetNS = [indicesVel[ii] for ii in dofFreeFSSetNS]
            # gldofSetFSNS = [indicesVel[ii] for ii in dofSetFSNS]
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            
            cornerCoords = self.dom.getCellCornersCoords(cell)
            locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)
            # print(self.comm.rank,'dofFree', dofFree)
            # print(self.comm.rank,'dofFreeFSSetNS', dofFreeFSSetNS )
            # print(self.comm.rank,'dofSetFSNS', dofSetFSNS)

            # # print(self.comm.rank , 'gldofFreeFSSetNS' ,gldofFreeFSSetNS) 
            # # print(self.comm.rank , 'gldofSetFSNS' ,gldofSetFSNS) 
            # print(self.comm.rank , 'gldof2beSet' ,gldof2beSet) 
            # print(self.comm.rank , 'gldofFree' ,gldofFree) 

            if nodeBCintersect:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)

                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    # setting values to 0 to allocate space see below
                    # setting to 1
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)

            # Elemental matrices assembled in the global distributed matrices
            self.mat.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.mat.K.setValues(indd, indd, 0, addv=True)

            self.mat.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.mat.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)
            
        self.mat.assembleAll()
        self.mat.setIndices2One(indices2one)

    def setUp(self, yamlInput):
        self.setUpGeneral(yamlInput['domain'])

        self.setUpBoundaryConditions(yamlInput)
        self.setUpEmptyMats()
        self.buildKLEMats()

    def setUpSolver(self, parameter_list):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError


OptDB = PETSc.Options()

comm = MPI.COMM_WORLD
yamlDir = OptDB.getString('-yaml')

with open(yamlDir) as f:
    yamlData = yaml.load(f, Loader=yaml.Loader)

fem = UniformFlow(comm)
fem.setUp(yamlData)