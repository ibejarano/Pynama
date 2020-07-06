from cases.base_problem import NoSlip
from common.nswalls import NoSlipWalls
import numpy as np

class Cavity(NoSlip):
    def setUp(self):
        super().setUp()
        self.collectCornerNodes()

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
        self.logger.info("Corner Nodes collected")

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

    def applyBoundaryConditions(self):
        self.vel.set(0.0)

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

        self.velFS.assemble()

    @staticmethod
    def VelCavity(coord,BoundaryConditions,dim,t=None):
        for bc in BoundaryConditions:
            if coord[bc[1]] == bc[0][1]:
                vel= bc[2]
                return vel
        vel=[0]*dim 
        return vel


