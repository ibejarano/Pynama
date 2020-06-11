import sys
import petsc4py
from math import pi, sin, cos, exp
petsc4py.init(sys.argv)

from cases.base_problem import BaseProblem
from solver.ksp_solver import KspSolver
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer

class TaylorGreen(BaseProblem):
    def setUpBoundaryConditions(self):
        self.dom.setLabelToBorders()
        self.tag2BCdict, self.node2tagdict = self.dom.readBoundaryCondition()

    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def computeInitialCondition(self, startTime):
        allNodes = self.dom.getAllNodes()
        fvort_coords = lambda coords: self.taylorGreenVortScalar(coords, t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort)

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
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.taylorGreenVelVec(coords, t=time)
        fvort_coords = lambda coords: self.taylorGreenVortScalar(coords, t=time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort)
        return exactVel, exactVort

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.taylorGreenVelVec(coords, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel)

    def solveKLETests(self, startTime=0.0, endTime=1.0, steps=10):
        times = np.linspace(startTime, endTime, steps)
        boundaryNodes = self.getBoundaryNodes()
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            self.mat.Curl.mult( exactVel , self.vort )
            self.viewer.saveVec(self.vel, timeStep=step)
            self.viewer.saveVec(self.vort, timeStep=step)
            self.viewer.saveVec(exactVel, timeStep=step)
            self.viewer.saveVec(exactVort, timeStep=step)
            self.viewer.saveStepInXML(step, time, vecs=[exactVel, exactVort, self.vel, self.vort])
        self.viewer.writeXmf(self.caseName)

    def solveKLE(self, time, vort):
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def getKLEError(self, times=None ,startTime=0.0, endTime=1.0, steps=10):
        try:
            assert times !=None
        except:
            times = np.arange(startTime, endTime, (endTime - startTime)/steps)

        boundaryNodes = self.getBoundaryNodes()
        errors = list()
        for time in times:
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            error = (exactVel - self.vel).norm(norm_type=2)
            errors.append(error)
        return errors

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
        return [vel[0], vel[1], 0]

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
        return [0,0,vort]