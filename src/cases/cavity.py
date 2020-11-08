from cases.base_problem import NoSlipFreeSlip
from common.nswalls import NoSlipWalls
import numpy as np
from math import sqrt, erf, exp, pi
import pandas as pd
import matplotlib.pyplot as plt
import csv


class Cavity(NoSlipFreeSlip):
    def setUp(self):
        super().setUp()
        self.collectCornerNodes()
        if self.dim==2:
            self.velFunction = self.flatplateVel
            self.vortFunction = self.flatplateVort
        else:
            self.velFunction = self.flatplateVel3d
            self.vortFunction = self.flatplateVort3d
            

    def collectCornerNodes(self):
        cornerNodes = set()
        allWalls = list(self.nsWalls.getWallsNames())
        while (len(allWalls) > 0 ):
            currentWall = allWalls.pop(0)
            currentNodes = self.dom.getBorderNodes(currentWall)
            currentNodes = set(currentNodes)
            for wall in allWalls:
                nodes = self.dom.getBorderNodes(wall)
                cornerNodes |= currentNodes & set(nodes)
        
        cornerNodes = list(cornerNodes)
        self.cornerDofs = [self.dim * node + dof for node in cornerNodes for dof in range(self.dim)]
        if not self.comm.rank:
            self.logger.info("Corner Nodes collected")

    def readBoundaryCondition(self,inputData):
        self.BoundaryCondition = list()
        try:
            self.nsWalls = NoSlipWalls(self.lower, self.upper, exclude=inputData['free-slip'].keys())
        except:
            self.nsWalls = NoSlipWalls(self.lower, self.upper)
        
        if 'no-slip' in inputData:
            for wallName, wallVelocity in inputData['no-slip'].items():
                self.nsWalls.setWallVelocity(wallName, wallVelocity)

    def setUpBoundaryConditions(self):
        self.dom.setLabelToBorders()

        bc = self.config.get("boundary-conditions")
        if 'free-slip' in bc:
            fsFaces = bc['free-slip'].keys()
        else:
            fsFaces = list()
        nsFaces = self.nsWalls.getWallsNames()
        self.dom.setBoundaryCondition(fsFaces, list(nsFaces))
        if not self.comm.rank:
            self.logger.info(f"Boundary Conditions setted up")

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        allNodes = self.dom.getAllNodes()
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu,t=startTime)
        self.vort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, self.vort, self.dim_w)
        self.vort.assemble()

    def applyBoundaryConditions(self, time):
        self.vel.set(0.0)
        # self.vort.view()
        if self.globalNodesDIR:
            fvel_coords = lambda coords: self.velFunction(coords, self.nu, t=time)
            fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t=time)
            self.vel = self.dom.applyFunctionVecToVec(self.globalNodesDIR, fvel_coords, self.vel, self.dim)
            self.vort = self.dom.applyFunctionVecToVec(self.globalNodesDIR, fvort_coords, self.vort, self.dim_w)

        wallsWithVel = self.nsWalls.getWallsWithVelocity()
        for wallName in wallsWithVel:
            nodes = self.dom.getBorderNodes(wallName)
            vel, velDofs = self.nsWalls.getWallVelocity(wallName)
            dofVelToSet = [node*self.dim + dof for node in nodes for dof in velDofs]
            self.vel.setValues(dofVelToSet, np.repeat(vel, len(nodes)))
        # set vel to zero in corner nodes
        self.vel.setValues(self.cornerDofs, np.repeat(0, len(self.cornerDofs)) )
        self.vel.assemble()

    def applyBoundaryConditionsFS(self):
        wallsWithVel = self.nsWalls.getWallsWithVelocity()
        staticWalls = self.nsWalls.getStaticWalls()

        for wallName in wallsWithVel:
            nodes = self.dom.getBorderNodes(wallName)
            vel, velDofs = self.nsWalls.getWallVelocity(wallName)
            dofVelToSet = [node*self.dim + dof for node in nodes for dof in velDofs]
            self.velFS.setValues(dofVelToSet, np.repeat(vel, len(nodes)))
        for staticWall in staticWalls:
            nodes = self.dom.getBorderNodes(staticWall)
            velDofs = self.nsWalls.getStaticDofsByName(staticWall)
            dofVelToSet = [node*self.dim + dof for node in nodes for dof in velDofs]
            self.velFS.setValues(dofVelToSet, np.repeat(0, len(nodes)*len(velDofs)))

        # set velfs to zero in corner nodes
        #self.velFS.setValues(self.cornerDofs, np.repeat(0, len(self.cornerDofs)) )
        self.velFS.assemble()
    
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
        #plt.savefig(f"Error-Velocidad-Log{self.nelem}-{self.ngl}")
        plt.figure(figsize=(10,10))
        plt.xlabel(r'tiempo')
        plt.ylabel(r'$||Error_{vel}||_{\infty}$')
        plt.plot(self.saveTime, self.saveError8 ,marker='o', markersize=3 ,color="b")
        plt.title(r'Error Infinito de la velocidad en el tiempo')
        plt.savefig(f"Error-Velocidad-NormaInfinito-placaplana{self.nelem}-{self.ngl}")
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'tiempo')
        # plt.ylabel(r'$||Error_{vel}||_{2}$')
        # plt.plot(self.saveTime, self.saveError2 ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Error de la velocidad en el tiempo')
        # plt.savefig(f"Error-Velocidad-NOLog{self.nelem}-{self.ngl}")
        # plt.figure(figsize=(10,10))
        # plt.xlabel(r'step')
        # plt.ylabel(r'tiempo')
        # plt.plot(self.saveStep, self.saveTime ,marker='o', markersize=3 ,color="b")
        # plt.title(r'Tiempo vs pasos')
        # plt.savefig(f"tiempos-pasos-{self.nelem}-{self.ngl}")
        with open ("data3dflateplate20.csv","a") as f:#save csv Error8 and time
             file_csv=csv.writer(f)
             file_csv.writerows([self.saveTime,self.saveError8])





    @staticmethod
    def flatplateVel(coord, nu , t=None):
        U_ref = 1
        vx = U_ref * erf(coord[1]/ sqrt(4*nu*t))
        vy = 0
        return [vx, vy]

    @staticmethod
    def flatplateVort(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        vort = (-2/(tau * sqrt(pi))) * exp(-(coord[1]/tau)**2)
        return [vort]

    @staticmethod
    def flatplateVel3d(coord, nu , t=None):
        U_ref = 1
        vx = U_ref * (1-erf(coord[1]/ sqrt(4*nu*t)))
        vy = 0
        vz = U_ref *(1- erf(coord[1]/ sqrt(4*nu*t)))
        return [vx, vy, vz]

    @staticmethod
    def flatplateVort3d(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        vort = (-2/(tau * sqrt(pi))) * exp(-(coord[1]/tau)**2)
        return [vort,0, -vort]