from cases.base_problem import NoSlipFreeSlip
from common.nswalls import NoSlipWalls
import numpy as np

class Cavity(NoSlipFreeSlip):
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

    def applyBoundaryConditions(self):
        self.vel.set(0.0)

        wallsWithVel = self.nsWalls.getWallsWithVelocity()
        for wallName in wallsWithVel:
            nodes = self.dom.getBorderNodes(wallName)
            vel, velDofs = self.nsWalls.getWallVelocity(wallName)
            dofVelToSet = [node*self.dim + dof for node in nodes for dof in velDofs]
            self.vel.setValues(dofVelToSet, np.repeat(vel, len(nodes)))

        # set vel to zero in corner nodes
        #self.vel.setValues(self.cornerDofs, np.repeat(0, len(self.cornerDofs)) )
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