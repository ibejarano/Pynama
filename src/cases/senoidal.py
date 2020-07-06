import sys
import petsc4py
from math import pi, sin, cos, exp
petsc4py.init(sys.argv)

from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer

class Senoidal(FreeSlip):
    def setUp(self):
        super().setUp()

        self.nu = self.mu / self.rho

        if self.dim == 2:
            self.taylorGreenVelFunction = self.taylorGreenVel_2D
            self.taylorGreenVortFunction = self.taylorGreenVort_2D
        else:
            self.taylorGreenVelFunction = self.taylorGreenVel_3D
            self.taylorGreenVortFunction = self.taylorGreenVort_3D

    def computeInitialCondition(self, startTime):
        allNodes = self.dom.getAllNodes()
        fvort_coords = lambda coords: self.taylorGreenVortFunction(coords, self.nu,)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)
        self.vort.assemble()

    def generateExactVecs(self):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactConv = exactVort.copy()
        exactDiff = exactVort.copy()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        exactConv.setName(f"{self.caseName}-exact-convective")
        exactDiff.setName(f"{self.caseName}-exact-diffusive")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.taylorGreenVelFunction(coords, self.nu)
        fvort_coords = lambda coords: self.taylorGreenVortFunction(coords, self.nu)
        fconv_coords = lambda coords: self.convective(coords, self.nu)
        fdiff_coords = lambda coords: self.difusive(coords, self.nu)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        exactConv = self.dom.applyFunctionVecToVec(allNodes, fconv_coords, exactConv, self.dim_w )
        exactDiff = self.dom.applyFunctionVecToVec(allNodes, fdiff_coords, exactDiff, self.dim_w )
        return exactVel, exactVort, exactConv, exactDiff

    def applyBoundaryConditions(self, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.taylorGreenVelFunction(coords, self.nu)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)
        self.vel.assemble()

    # Se llama solve KLE pero es de los operators
    def solveKLETests(self):
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(boundaryNodes)
        step = 0
        exactVel, exactVort, exactConv, exactDiff = self.generateExactVecs()
        self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
        convectivo = self.getconvectivo(exactVel, exactConv)
        convectivo.setName("convectivo")
        difusivo = self.getDifusivo(exactVel, exactDiff)
        difusivo.setName("difusivo")
        self.operator.Curl.mult(exactVel, self.vort)
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveVec(exactVel, timeStep=step)
        self.viewer.saveVec(convectivo, timeStep=step)
        self.viewer.saveVec(exactVort, timeStep=step)
        self.viewer.saveVec(exactConv, timeStep=step)
        self.viewer.saveVec(exactDiff, timeStep=step)
        self.viewer.saveVec(difusivo, timeStep=step)
        self.viewer.saveStepInXML(step, time=0.0, vecs=[exactVel, exactVort, exactConv, exactDiff, self.vort, self.vel, convectivo, difusivo])
        self.viewer.writeXmf(self.caseName)
        self.logger.info("Operatores Tests")

    def getconvectivo(self,exactVel, exactConv):
        convectivo = exactConv.copy()
        sK, eK = self.mat.K.getOwnershipRange()
        for indN in range(sK, eK, self.dim):
            indicesVV = [indN * self.dim_s / self.dim + d
                         for d in range(self.dim_s)]
            VelN = exactVel.getValues([indN + d for d in range(self.dim)])
            if self.dim==2:
                VValues = [VelN[0] ** 2, VelN[0] * VelN[1], VelN[1] ** 2]
            else:
                raise Exception("Wrong dim")
            self._VtensV.setValues(indicesVV, VValues, False)
        aux=self.vel.copy()
        self.operator.DivSrT.mult(self._VtensV, aux)
        self.operator.Curl.mult(aux,convectivo)
        return convectivo

    def getDifusivo(self,exactVel, exactDiff):
        difusivo = exactDiff.copy()
        self.operator.SrT.mult(exactVel, self._Aux1)
        aux=self.vel.copy()
        self._Aux1 *= (2.0 * self.mu)
        self.operator.DivSrT.mult(self._Aux1, aux)
        aux.scale(1/self.rho)
        self.operator.Curl.mult(aux,difusivo)
        return difusivo

    @staticmethod
    def taylorGreenVel_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vel = [sin(x_), sin(y_)]
        return [vel[0], vel[1]]

    @staticmethod
    def taylorGreenVort_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vort = Wref_x * pi *  cos(y_) - Wref_y * pi * cos(x_)
        return [vort]

    @staticmethod
    def convective(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        conv = ((Wref_y * pi)**2 - (Wref_x * pi )**2) * sin(x_) * sin(y_)
        return [conv]

    @staticmethod
    def difusive(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        tmp1 = -(Wref_x *pi)**3 * cos(y_) 
        tmp2 = (Wref_y *pi)**3 * cos(x_) 
        return [tmp1 + tmp2]

    @staticmethod
    def taylorGreenVel_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return [vel[0], vel[1], 0]

    @staticmethod
    def taylorGreenVort_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return [0,0,vort]