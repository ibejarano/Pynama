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
from matrices.mat_generator import Mat

class Cavity(NoSlip):
    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        fakeConectMat = self.dom.getDMConectivityMat()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()
        self.mat.createEmptyKLEMatsNS(fakeConectMat, globalIndicesDIR, self.node2tagdict,createOperators=True)

    def computeInitialCondition(self, startTime):
        allNodes = self.dom.getAllNodes()
        fvort_coords = lambda coords: self.VortCavity(coords, t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)



    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.VelCavity(coords, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)


    @staticmethod
    def VelCavity(coord,t=None):
        for bc in self.applyBoundaryConditions:
            if coord[bc[1]] == bc[0][1]:
                vel= bc[2] 
        return vel

    @staticmethod
    def VortCavity_2D(coord, t=None):
        return [0]*self.dim_w