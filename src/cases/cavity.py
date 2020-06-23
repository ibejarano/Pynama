from cases.base_problem import NoSlip
from common.nswalls import NoSlipWalls
import numpy as np

class Cavity(NoSlip):
    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def readBoundaryCondition(self,inputData):
        bcdict = inputData['border-name']
        wallsWithVelocity = inputData['no-slip']
        self.BoundaryCondition = list()

        for bc in bcdict.keys():
            if bc[:5]=="upper":
                self.BoundaryCondition.append((self.upper,bcdict[bc]["coord"],bcdict[bc]["vel"]))
            if bc[:5]=="lower":
                self.BoundaryCondition.append((self.lower,bcdict[bc]["coord"],bcdict[bc]["vel"]))

        # Otra alternativa
        self.nsWalls = NoSlipWalls(self.lower, self.upper)
        for wallName, wallVelocity in wallsWithVelocity.items():
            self.nsWalls.setWallVelocity(wallName, wallVelocity)


    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)

        wallsWithVel = self.nsWalls.getWallsWithVelocity()
        for wallName in wallsWithVel:
            nodes = self.dom.getBorderNodes(wallName)
            vel, velDofs = self.nsWalls.getWallVelocity(wallName)
            dofVelToSet = [node*self.dim + dof for node in nodes for dof in velDofs]
            self.vel.setValues(dofVelToSet, np.repeat(vel, len(nodes)))

        # fvel_coords = lambda coords: self.VelCavity(coords,self.BoundaryCondition,self.dim, t=time)
        # self.vel = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.vel, self.dim)


    def applyBoundaryConditionsFS(self, time, bcNodes):
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
            self.velFS.setValues(dofVelToSet, np.repeat(0, len(nodes)))

        # # TODO Set the tang of walls without vel to 0

        # fvel_coords = lambda coords: self.VelCavity(coords,self.BoundaryCondition,self.dim, t=time)
        # self.velFS = self.dom.applyFunctionVecToVec(bcNodes, fvel_coords, self.velFS, self.dim)
    

    @staticmethod
    def VelCavity(coord,BoundaryConditions,dim,t=None):
        for bc in BoundaryConditions:
            if coord[bc[1]] == bc[0][1]:
                vel= bc[2]
                return vel
        vel=[0]*dim 
        return vel