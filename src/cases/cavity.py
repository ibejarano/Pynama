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
        allNodes = self.dom.getAllNodes()
        fvort_coords = lambda coords: self.VortCavity(coords,self.dim_w, t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)



    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.VelCavity(coords,self.applyBoundaryConditions,self.dim, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)


    @staticmethod
    def VelCavity(coord,applyBoundaryConditions,dim,t=None):
        for bc in applyBoundaryConditions:
            if coord[bc[1]] == bc[0][1]:
                vel= bc[2]
                return vel
        vel=[0]*dim 
        return vel

    @staticmethod
    def VortCavity(coord,dim_w ,t=None):
        return [0]*dim_w