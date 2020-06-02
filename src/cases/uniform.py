import sys
import petsc4py

petsc4py.init(sys.argv)

from cases.base_problem import BaseProblem
from matrices.mat_generator import Mat
from solver.ksp_solver import KspSolver
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer

class UniformFlow(BaseProblem):
    def __init__(self):
        super().__init__()

        # OptDB = PETSc.Options()
        # yamlDir = OptDB.getString('-yaml')

        with open('src/cases/uniform.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)

        self.setUp(yamlData)

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
                # print("gldof2beSet {}".format(gldof2beSet))
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)

                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    # setting values to 0 to allocate space see below
                    # setting to 1
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)

            # print('gldoffree : {}'.format(gldofFree))
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

    def setUpSolver(self):
        self.solver = KspSolver()
        self.solver.createSolver(self.mat.K, self.comm)
        self.vel = self.mat.K.createVecRight()
        self.vel.setName("velocity")
        self.vort = self.mat.Rw.createVecRight()
        self.vort.setName("vorticity")
        self.vort.set(0.0)
        # self.vel.view()
        # self.vort.view()
        boundaryNodes = self.getBoundaryNodes()
        boundaryVelocityIndex = self.dom.getVelocityIndex(boundaryNodes)
        boundaryVelocityValues = [1 , 0] * len(boundaryNodes)
        
        self.vel.setValues(boundaryVelocityIndex, boundaryVelocityValues , addv=False)
        # self.vel.view()
        self.vel.assemble()

    def getBoundaryNodes(self):
        """ IS: Index Set """
        nodesSet = set()
        IS =self.dom.dm.getStratumIS('marco', 0)
        entidades = IS.getIndices()
        for entity in entidades:
            nodes = self.dom.getGlobalNodesFromCell(entity, False)
            # print("[{}]".format(self.comm.rank), "entity: {}".format(entity)  , "nodos: {}".format(nodes))
            nodesSet |= set(nodes)
        
        return list(nodesSet)

    def generateExactVel(self):
        exactVel = self.mat.K.createVecRight()
        totalVel = exactVel.getSize()
        totalNodes = range(int(totalVel/self.dim))
        exactValues = [ 1, 0 ] * len(totalNodes)
        indexVel = self.dom.getVelocityIndex(totalNodes)
        exactVel.setValues( indexVel, exactValues, addv=False )
        exactVel.assemble()
        return exactVel

    def solve(self):
        self.solver( self.mat.Rw * self.vort + self.mat.Krhs * self.vel , self.vel)
        self.viewer.saveVec(self.vel, timeStep=1)
        self.viewer.saveStepInXML(self.vel, 1, 0.001)
        self.viewer.writeXmf("uniform-flow")

        # self.vel.view()

