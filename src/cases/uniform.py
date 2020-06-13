import sys
import petsc4py
petsc4py.init(sys.argv)

from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer

class UniformFlow(FreeSlip):
    def setUp(self):
        self.setUpGeneral()

        self.cteValue = [1,0]

        if self.dim == 2:
            print("dim 2")
        else:
            print("unknown")
            raise Exception("wrong dim 3 or another")

        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        self.vel = self.dom.applyValuesToVec(bcNodes, self.cteValue, self.vel)

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

    def generateExactVecs(self, time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        exactVel = self.dom.applyValuesToVec(allNodes, self.cteValue, exactVel)
        exactVort.set(0.0)
        return exactVel, exactVort