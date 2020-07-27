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


class TaylorGreen(FreeSlip):
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
        fvort_coords = lambda coords: self.taylorGreenVortFunction(coords, self.nu,t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)
        self.vort.assemble()

    def generateExactVecs(self, time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.taylorGreenVelFunction(coords, self.nu, t=time)
        fvort_coords = lambda coords: self.taylorGreenVortFunction(coords, self.nu, t=time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        return exactVel, exactVort

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.taylorGreenVelFunction(coords, self.nu, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)
        self.vel.assemble()

    def solveKLETests(self, startTime=0.0, endTime=1.0, steps=10):
        times = np.linspace(startTime, endTime, steps)
        boundaryNodes = self.getBoundaryNodes()
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            self.operator.Curl.mult( exactVel , self.vort )
            self.viewer.saveVec(self.vel, timeStep=step)
            self.viewer.saveVec(self.vort, timeStep=step)
            self.viewer.saveVec(exactVel, timeStep=step)
            self.viewer.saveVec(exactVort, timeStep=step)
            self.viewer.saveStepInXML(step, time, vecs=[exactVel, exactVort, self.vort, self.vel])
        self.viewer.writeXmf(self.caseName)


    def generateExactOperVecs(self,time):
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
        fvel_coords = lambda coords: self.taylorGreenVelFunction(coords, self.nu, t = time)
        fvort_coords = lambda coords: self.taylorGreenVortFunction(coords, self.nu, t = time)
        fconv_coords = lambda coords: self.taylorGreen3dConvective(coords, self.nu, t = time)
        fdiff_coords = lambda coords: self.taylorGreen3dDiffusive(coords, self.nu, t = time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        exactConv = self.dom.applyFunctionVecToVec(allNodes, fconv_coords, exactConv, self.dim_w )
        exactDiff = self.dom.applyFunctionVecToVec(allNodes, fdiff_coords, exactDiff, self.dim_w )
        return exactVel, exactVort, exactConv, exactDiff


    # Se llama solve KLE pero es de los operators
    def OperatorsTests(self):
        time = 0.0
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        step = 0
        exactVel, exactVort, exactConv, exactDiff = self.generateExactOperVecs(time)
        self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
        convective = self.getConvective(exactVel, exactConv)
        convective.setName("convective")
        diffusive = self.getDiffusive(exactVel, exactDiff)
        diffusive.setName("diffusive")
        self.operator.Curl.mult(exactVel, self.vort)
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveVec(exactVel, timeStep=step)
        self.viewer.saveVec(convective, timeStep=step)
        self.viewer.saveVec(exactVort, timeStep=step)
        self.viewer.saveVec(exactConv, timeStep=step)
        self.viewer.saveVec(exactDiff, timeStep=step)
        self.viewer.saveVec(diffusive, timeStep=step)
        self.viewer.saveStepInXML(step, time=0.0, vecs=[exactVel, exactVort, exactConv, exactDiff, self.vort, self.vel, convective, diffusive])
        self.viewer.writeXmf(self.caseName)
        errorConv = (convective - exactConv).norm(norm_type=2)
        errorDiff = (diffusive - exactDiff).norm(norm_type=2)
        errorCurl = (self.vort - exactVort).norm(norm_type=2)
        self.logger.info("Operatores Tests")
        return errorConv, errorDiff, errorCurl

    def getConvective(self,exactVel, exactConv):
        convective = exactConv.copy()
        sK, eK = self.mat.K.getOwnershipRange()
        for indN in range(sK, eK, self.dim):
            indicesVV = [indN * self.dim_s / self.dim + d
                         for d in range(self.dim_s)]
            VelN = exactVel.getValues([indN + d for d in range(self.dim)])
            if self.dim==2:
                VValues = [VelN[0] ** 2, VelN[0] * VelN[1], VelN[1] ** 2]
            elif self.dim==3:
                VValues = [VelN[0] ** 2, VelN[0] * VelN[1] ,VelN[1] ** 2 , VelN[1] * VelN[2] , VelN[2] **2 , VelN[2] *VelN[0]]
            self._VtensV.setValues(indicesVV, VValues, False)
        aux=self.vel.copy()
        self.operator.DivSrT.mult(self._VtensV, aux)
        self.operator.Curl.mult(aux,convective)
        return convective

    def getDiffusive(self,exactVel, exactDiff):
        diffusive = exactDiff.copy()
        self.operator.SrT.mult(exactVel, self._Aux1)
        aux=self.vel.copy()
        self._Aux1 *= (2.0 * self.mu)
        self.operator.DivSrT.mult(self._Aux1, aux)
        aux.scale(1/self.rho)
        self.operator.Curl.mult(aux,diffusive)
        return diffusive

    @staticmethod
    def taylorGreenVel_2D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return [vel[0], vel[1]]

    @staticmethod
    def taylorGreenVort_2D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return [vort]

    @staticmethod
    def taylorGreenVel_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Lz = 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vel = [cos(x_) * sin(y_) *sin(z_)*Lx* expon, sin(x_) * cos(y_) *sin(z_)*Ly *expon,-2*sin(x_)* sin(y_) * cos(z_) *Lz* expon]
        return vel

    @staticmethod
    def taylorGreenVort_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vort = [-2 * pi * (Ly / Lz + 2* Lz / Ly) * sin(x_) * cos(y_) *cos(z_)* expon,2 * pi * (Lx / Lz + 2* Lz / Lx) * cos(x_) * sin(y_) *cos(z_)* expon,2 * pi * (Ly / Lx - Lx / Ly) * cos(x_) * cos(y_) *sin(z_)* expon]
        return vort

    @staticmethod
    def taylorGreen3dConvective(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        conv = [-2*(2*Lz/Ly+Ly/Lz)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(z_)*cos(z_), \
            2*(2*Lz/Lx+Lx/Lz)*( 2 * pi *expon)**2*sin(x_)*cos(x_)*sin(z_)*cos(z_),\
                -2*(2*Lx/Ly-2*Ly/Lx)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(x_)*cos(x_)]
        return conv

    @staticmethod
    def taylorGreen3dDiffusive(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        diff = [(2*pi)**3*expon*sin(x_)*cos(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Ly)+Lz/(Ly*Ly*Ly)+Lz/(Lz*Lz*Ly))+Ly/(Lx*Lx*Lz)+Ly/(Ly*Ly*Lz)+Ly/(Lz*Lz*Lz)),\
            (2*pi)**3*expon*cos(x_)*sin(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Lx)+Lz/(Ly*Ly*Lx)+Lz/(Lz*Lz*Lx))-Lx/(Lx*Lx*Lz)-Lx/(Ly*Ly*Lz)-Lx/(Lz*Lz*Lz)),\
            (2*pi)**3*expon*cos(x_)*cos(y_)*sin(z_)*(Lx/(Lx*Lx*Ly)+Lx/(Ly*Ly*Ly)+Lx/(Lz*Lz*Ly)-Ly/(Lx*Lx*Lx)-Ly/(Ly*Ly*Lx)-Ly/(Lz*Lz*Lx))]
        return diff