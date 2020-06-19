import sys
import petsc4py
from math import pi, sin, cos, exp
petsc4py.init(sys.argv)
from cases.base_problem import NoSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer


class Cavity(NoSlip):
    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.VelCavity(coords,self.BoundaryCondition,self.dim, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)


    def applyBoundaryConditionsFS(self, time, bcNodes):
        fvel_coords = lambda coords: self.VelCavity(coords,self.BoundaryCondition,self.dim, t=time)
        self.velFS = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.velFS, self.dim)
    

    @staticmethod
    def VelCavity(coord,BoundaryConditions,dim,t=None):
        for bc in BoundaryConditions:
            if coord[bc[1]] == bc[0][1]:
                vel= bc[2]
                return vel
        vel=[0]*dim 
        return vel

