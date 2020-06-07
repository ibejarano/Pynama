import sys
import petsc4py
from math import pi, sin, cos, exp
petsc4py.init(sys.argv)

from cases.base_problem import BaseProblem
from matrices.mat_generator import Mat
from solver.ksp_solver import KspSolver
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer

class TaylorGreen(BaseProblem):
    def __init__(self):
        super().__init__()

        with open('src/cases/taylor-green.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)

        self.setUp(yamlData)

    def setUpBoundaryConditions(self, inputData):
        self.dom.setLabelToBorders()
        self.tag2BCdict, self.node2tagdict = self.dom.readBoundaryCondition(inputData)

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        fakeConectMat = self.dom.getDMConectivityMat()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()
        self.mat.createEmptyKLEMats(fakeConectMat, globalIndicesDIR)

    def buildKLEMats(self):
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        indices2onefs = set()  # idem for FS solution
        boundaryNodes = set(self.node2tagdict.keys())
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)

        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            self.logger.debug("DMPlex cell: %s", cell)

            nodes = self.dom.getGlobalNodesFromCell(cell, shared=False)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
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
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            

            if nodeBCintersect:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)

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
        boundaryNodes = self.getBoundaryNodes()
        boundaryVelocityIndex = self.dom.getVelocityIndex(boundaryNodes)
        boundaryVelocityValues = [1 , 0] * len(boundaryNodes)
        self.vel.setValues(boundaryVelocityIndex, boundaryVelocityValues , addv=False)
        self.vel.assemble()

    def getBoundaryNodes(self):
        """ IS: Index Set """
        nodesSet = set()
        IS =self.dom.getStratumIS('marco', 0)
        entidades = IS.getIndices()
        for entity in entidades:
            nodes = self.dom.getGlobalNodesFromCell(entity, False)
            nodesSet |= set(nodes)
        return list(nodesSet)

    def generateExactVecs(self, time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName("tg-exact-vel")
        exactVort.setName("tg-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.taylorGreenVelVec(coords, t=time)
        fvort_coords = lambda coords: self.taylorGreenVortScalar(coords, t=time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel)
        exactVort = self.dom.applyFunctionScalarToVec(allNodes, fvort_coords, exactVort)
        return exactVel, exactVort

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.taylorGreenVelVec(coords, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel)

    def solve(self):
        startTime = 0.0
        endTime = 0.03
        steps = 100
        times = np.arange(startTime, endTime, (endTime - startTime)/steps)
        boundaryNodes = self.getBoundaryNodes()
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            self.viewer.saveVec(self.vel, timeStep=step)
            self.viewer.saveVec(exactVel, timeStep=step)
            self.viewer.saveVec(exactVort, timeStep=step)
            self.viewer.saveStepInXML(step, time, vecs=[exactVel, exactVort, self.vel])
        self.viewer.writeXmf("taylor-green")

    @staticmethod
    def taylorGreenVelVec(coord, t=None):
        Lx= 1
        Ly= 1
        nu = 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return vel

    @staticmethod
    def taylorGreenVortScalar(coord, t=None):
        Lx= 1
        Ly= 1
        nu = 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return vort