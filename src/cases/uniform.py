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

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)

    # def solveKLETests(self, startTime=0.0, endTime=1.0, steps=10):
    #     times = np.linspace(startTime, endTime, steps)
    #     for step,time in enumerate(times):
    #         exactVel, exactVort = self.generateExactVecs(time)
    #         self.applyBoundaryConditions(time)
    #         self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
    #         self.mat.Curl.mult( exactVel , self.vort )
    #         self.viewer.saveVec(self.vel, timeStep=step)
    #         self.viewer.saveVec(self.vort, timeStep=step)
    #         self.viewer.saveVec(exactVel, timeStep=step)
    #         self.viewer.saveVec(exactVort, timeStep=step)
    #         self.viewer.saveStepInXML(step, time, vecs=[exactVel, exactVort, self.vel, self.vort])
    #     self.viewer.writeXmf(self.caseName)

    def generateExactVecs(self, time=None):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        arr = np.tile([1,0], len(allNodes))
        # print(arr)
        print(self.dim)
        inds = [i*self.dim + dof for i in allNodes for dof in range(self.dim)]
        exactVel.setValues(inds, arr, addv=False)
        # exactVel = self.dom.applyValuesToVec(allNodes, [1,0], exactVel)
        exactVort.set(0.0)
        return exactVel, exactVort