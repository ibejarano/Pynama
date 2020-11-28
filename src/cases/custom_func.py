import sys
import petsc4py
from math import pi, sin, cos, exp, erf, sqrt
petsc4py.init(sys.argv)
import pandas as pd
import matplotlib.pyplot as plt
from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer
from viewer.plotter import Plotter
import csv


class CustomFuncCase(FreeSlip):
    # @profile
    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

        self.nu = self.mu / self.rho

        if self.case == 'taylor-green':
            if self.dim == 2:
                self.velFunction = self.taylorGreenVel_2D
                self.vortFunction = self.taylorGreenVort_2D
            else:
                self.velFunction = self.taylorGreenVel_3D
                self.vortFunction = self.taylorGreenVort_3D
                self.diffusiveFunction = self.taylorGreen3dDiffusive
                self.convectiveFunction = self.taylorGreen3dConvective

        elif self.case == 'taylor-green3dN':
                self.velFunction = self.taylorGreenVel_3DN
                self.vortFunction = self.taylorGreenVort_3DN
                self.diffusiveFunction = self.taylorGreen3DNDiffusive
                self.convectiveFunction = self.taylorGreen3DNConvective

        elif self.case == 'taylor-green2d-3d':
            self.velFunction = self.taylorGreenVel_2D_3D
            self.vortFunction = self.taylorGreenVort_2D_3D
        elif self.case == 'taylor-green2d-3d-nx':
            self.velFunction = self.taylorGreenVel_2D_3D_Nx
            self.vortFunction = self.taylorGreenVort_2D_3D_Nx
        elif self.case == 'taylor-green2d-3d-ny':
            self.velFunction = self.taylorGreenVel_2D_3D_Ny
            self.vortFunction = self.taylorGreenVort_2D_3D_Ny
        elif self.case == 'senoidal':
            if self.dim == 2:
                self.velFunction = self.senoidalVel_2D
                self.vortFunction = self.senoidalVort_2D
                self.diffusiveFunction = self.senoidalDiffusive
                self.convectiveFunction = self.senoidalConvective
            else:
                raise Exception("not defined func")
        elif self.case == 'flat-plate':
            if self.dim == 2:
                self.velFunction = self.flatplateVel
                self.vortFunction = self.flatplateVort
                self.diffusiveFunction = self.flatplateDiffusive
                self.convectiveFunction = self.flatplateConvective
            else:
                raise Exception("not implemented func for dim 3")


    def computeInitialCondition(self, startTime):
        allNodes = self.dom.getAllNodes()
        #fvel_coords = lambda coords: self.velFunction(coords, self.nu, t=startTime)
        #vel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, self.vel, self.dim)
        #self.vort = self.operator.Curl * vel
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu,t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)
        self.vort.assemble()



    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        fvel_coords = lambda coords: self.velFunction(coords, self.nu, t=time)
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t=time)
        self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)
        self.vort = self.dom.applyFunctionVecToVec(bcNodes, fvort_coords, self.vort, self.dim_w)
        self.vel.assemble()
        self.vort.assemble()

    def solveKLETests(self, steps=10):
        self.logger.info("Running KLE Tests")
        startTime = self.ts.getTime()
        endTime = self.ts.getMaxTime()
        times = np.linspace(startTime, endTime, steps)
        viscousTimes=[0.01,0.4,0.9]
        times = [(tau**2)/(4*self.nu) for tau in viscousTimes]
        nodesToPlot, coords = self.dom.getNodesOverline("x", 0.5)
        boundaryNodes = self.getBoundaryNodes()
        plotter = Plotter(r"$\frac{u}{U}$" , r"$\frac{y}{Y}$")
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            self.operator.Curl.mult( exactVel , self.vort )
            self.viewer.saveData(step, time, self.vel, self.vort, exactVel, exactVort)
            exact_x , _ = self.dom.getVecArrayFromNodes(exactVel, nodesToPlot)
            calc_x, _ = self.dom.getVecArrayFromNodes(self.vel, nodesToPlot)
            plotter.updatePlot(exact_x, [{"name": fr"$\tau$ = ${viscousTimes[step]}$" ,"data":coords}], step )
            # plotter.scatter(calc_x , coords, "calc")
            # plotter.plt.pause(0.001)
            self.logger.info(f"Saving time: {time:.1f} | Step: {step}")
        plotter.plt.legend()
        plotter.show()
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
        fvel_coords = lambda coords: self.velFunction(coords, self.nu, t = time)
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t = time)
        fconv_coords = lambda coords: self.convectiveFunction(coords, self.nu, t = time)
        fdiff_coords = lambda coords: self.diffusiveFunction(coords, self.nu, t = time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        exactConv = self.dom.applyFunctionVecToVec(allNodes, fconv_coords, exactConv, self.dim_w )
        exactDiff = self.dom.applyFunctionVecToVec(allNodes, fdiff_coords, exactDiff, self.dim_w )
        return exactVel, exactVort, exactConv, exactDiff

    def OperatorsTests(self):
        time = 1
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        step = 0
        exactVel, exactVort, exactConv, exactDiff = self.generateExactOperVecs(time)
        #exactDiff.scale(2*self.mu)
        self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
        convective = self.getConvective(exactVel, exactConv)
        convective.setName("convective")
        diffusive = self.getDiffusive(exactVel, exactDiff)
        diffusive.setName("diffusive")
        self.operator.Curl.mult(exactVel, self.vort)
        self.viewer.saveData(step, time, self.vel, self.vort, exactVel, exactVort,exactConv,exactDiff,convective,diffusive )
        self.viewer.writeXmf(self.caseName)
        self.operator.weigCurl.reciprocal()
        err = convective - exactConv
        errorConv = sqrt((err * err ).dot(self.operator.weigCurl))
        err = diffusive - exactDiff
        errorDiff = sqrt((err * err ).dot(self.operator.weigCurl))
        err = self.vort - exactVort
        errorCurl = sqrt((err * err ).dot(self.operator.weigCurl))
        self.logger.info("Operatores Tests")
        return errorConv, errorDiff, errorCurl

    def generateExactVecs(self, time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.velFunction(coords, self.nu, t=time)
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t=time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        return exactVel, exactVort

    def getChartKLE(self):
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'tiempo')
        # plt.ylabel(r'$||Error_{vel}||_{2}$')
        # plt.loglog(self.saveTime, self.saveError2 ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Error norma 2 de la velocidad en el tiempo')
        # plt.savefig(f"Error-Velocidad-Log")
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'tiempo')
        # plt.ylabel(r'$||Error_{vel}||_{\infty}$')
        # plt.plot(self.saveTime, self.saveError8 ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Error Infinito de la velocidad en el tiempo')
        # plt.savefig(f"Error-Velocidad-NormaInfinito")
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'tiempo')
        # plt.ylabel(r'$||Error_{vel}||_{2}$')
        # plt.plot(self.saveTime, self.saveError2 ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Error de la velocidad en el tiempo')
        # plt.savefig(f"Error-Velocidad-NOLog")
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'step')
        # plt.ylabel(r'tiempo')
        # plt.plot(self.saveStep, self.saveTime ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Tiempo vs pasos')
        # plt.savefig(f"tiempos-pasos")
        with open ("large.csv","a") as f:#save csv Error8 and time
             file_csv=csv.writer(f)
             file_csv.writerows([self.saveTime,self.saveError8])


    def getConvective(self,exactVel, exactConv):
        convective = exactConv.copy()
        self.computeVtensV(velocity=exactVel)
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
    def taylorGreenVel_2D_3D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return [vel[0], vel[1],0]

    @staticmethod
    def taylorGreenVort_2D_3D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return [0,0,vort]
    



    @staticmethod
    def taylorGreenVel_2D_3D_Nx(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        y_ = 2 * pi * coord[1] / Lx
        z_ = 2 * pi * coord[2] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(y_) * sin(z_) * expon, -sin(y_) * cos(z_) * expon]
        return [0,vel[0], vel[1]]

    @staticmethod
    def taylorGreenVort_2D_3D_Nx(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        y_ = 2 * pi * coord[1] / Lx
        z_ = 2 * pi * coord[2] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(y_) * cos(z_) * expon
        return [vort,0,0]

    @staticmethod
    def taylorGreenVel_2D_3D_Ny(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        z_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(z_) * expon, -sin(x_) * cos(z_) * expon]
        return [vel[0],0, vel[1]]

    @staticmethod
    def taylorGreenVort_2D_3D_Ny(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        z_ = 2 * pi * coord[2] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = 2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(z_) * expon
        return [0,vort,0]

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
                2*(2*Lx/Ly-2*Ly/Lx)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(x_)*cos(x_)]
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
        expon = Uref *nu * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        diff = [(2*pi)**3*expon*sin(x_)*cos(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Ly)+Lz/(Ly*Ly*Ly)+Lz/(Lz*Lz*Ly))+Ly/(Lx*Lx*Lz)+Ly/(Ly*Ly*Lz)+Ly/(Lz*Lz*Lz)),\
            -(2*pi)**3*expon*cos(x_)*sin(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Lx)+Lz/(Ly*Ly*Lx)+Lz/(Lz*Lz*Lx))+Lx/(Lx*Lx*Lz)+Lx/(Ly*Ly*Lz)+Lx/(Lz*Lz*Lz)),\
             (2*pi)**3*expon*cos(x_)*cos(y_)*sin(z_)*(Lx/(Lx*Lx*Ly)+Lx/(Ly*Ly*Ly)+Lx/(Lz*Lz*Ly)-Ly/(Lx*Lx*Lx)-Ly/(Ly*Ly*Lx)-Ly/(Lz*Lz*Lx))]
        return diff

    @staticmethod
    def taylorGreenVel_3DN(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Lz = 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vel = [cos(x_) * sin(y_) *sin(z_)*Lx* expon, -sin(x_) * cos(y_) *sin(z_)*Ly *expon,0]
        return vel

    @staticmethod
    def taylorGreenVort_3DN(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vort = [2 * pi * (Ly / Lz ) * sin(x_) * cos(y_) *cos(z_)* expon,2 * pi * (Lx / Lz ) * cos(x_) * sin(y_) *cos(z_)* expon,- 2 * pi * (Ly / Lx + Lx / Ly) * cos(x_) * cos(y_) *sin(z_)* expon]
        return vort

    @staticmethod
    def taylorGreen3DNConvective(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        conv = [2*(Ly/Lz)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(z_)*cos(z_), \
                 -2*(Lx/Lz)*( 2 * pi *expon)**2*sin(x_)*cos(x_)*sin(z_)*cos(z_),\
                0]
        return conv

    @staticmethod
    def taylorGreen3DNDiffusive(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref *nu * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        diff = [-(2*pi)**3*expon*sin(x_)*cos(y_)*cos(z_)*(Ly/(Lx*Lx*Lz)+Ly/(Ly*Ly*Lz)+Ly/(Lz*Lz*Lz)),\
            -(2*pi)**3*expon*cos(x_)*sin(y_)*cos(z_)*(Lx/(Lx*Lx*Lz)+Lx/(Ly*Ly*Lz)+Lx/(Lz*Lz*Lz)),\
             (2*pi)**3*expon*cos(x_)*cos(y_)*sin(z_)*(Lx/(Lx*Lx*Ly)+Lx/(Ly*Ly*Ly)+Lx/(Lz*Lz*Ly)+Ly/(Lx*Lx*Lx)+Ly/(Ly*Ly*Lx)+Ly/(Lz*Lz*Lx))]
        return diff

    @staticmethod
    def senoidalVel_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vel = [sin(x_), sin(y_)]
        return [vel[0], vel[1]]

    @staticmethod
    def senoidalVort_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vort = Wref_x * pi *  cos(y_) - Wref_y * pi * cos(x_)
        return [vort]

    @staticmethod
    def senoidalConvective(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        conv = ((Wref_y * pi)**2 - (Wref_x * pi )**2) * sin(x_) * sin(y_)
        return [conv]

    @staticmethod
    def senoidalDiffusive(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        tmp1 = -(Wref_x *pi)**3 * cos(y_) 
        tmp2 = (Wref_y *pi)**3 * cos(x_) 
        return [tmp1 + tmp2]

    @staticmethod
    def flatplateVel(coord, nu , t=None):
        U_ref = 1
        vx = U_ref * erf(coord[1]/ sqrt(4*nu*t))
        vy = 1
        return [vx, vy]

    @staticmethod
    def flatplateVort(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        vort = (-2/(tau * sqrt(pi))) * exp(-(coord[1]/tau)**2)
        return [vort]

    @staticmethod
    def flatplateConvective(coord, nu, t=None):
        c = 1
        tau = sqrt(4*nu*t)
        alpha = 4 * c * coord[1] / ( sqrt(pi) * tau**3 )
        convective = alpha * exp( -(coord[1]/tau)**2 )
        return [convective]

    @staticmethod
    def flatplateDiffusive(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        alpha = 4 / (sqrt(pi)* tau**3)
        beta = ( 1 - 2 * coord[1]**2 / tau**2 )
        diffusive = nu * alpha * beta * exp( -(coord[1]/tau)**2 )
        return [diffusive]