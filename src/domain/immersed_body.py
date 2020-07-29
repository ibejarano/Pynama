import numpy as np
from math import sin, cos , pi , sqrt, ceil, floor
from petsc4py import PETSc
import logging


class ImmersedBody:
    def __init__(self, vel=[0,0], center=[0,0]):
        self.dirac = fourGrid
        self.__centerDisplacement = center
        self.__dl = None
        self.__vel = vel
        self.logger = logging.getLogger("Body Immersed")

    def setVelRef(self, vel):
        self.__Uref = vel

    def view(self):
        self.logger.info(f"Arc len: {self.__dl} | Dirac Type: {self.dirac.__name__} | Vel Fluid Reference: {self.__Uref} ")

    def viewState(self):
        self.logger.info(f"Body vel: {self.__vel} | Body position {self.__centerDisplacement}")
    
    def setUpDimensions(self):
        self.firstNode, self.lastNode = self.__dom.getHeightStratum(1)
        self.coordinates = self.__dom.getCoordinatesLocal()
        self.coordSection = self.__dom.getCoordinateSection()
        self.__lagNodes = [0]*(self.lastNode - self.firstNode)

    def setEulerNodes(self, lagPoi, NNZNodes):
        self.__lagNodes[lagPoi] = NNZNodes

    def generateDMPlex(self, dh, dim=1):
        coords, cone ,dl = self.generateBody(dh)
        self.__dl = dl
        self.__L = self.getLong()
        self.__dom = PETSc.DMPlex().createFromCellList(dim, cone,coords)
        self.setUpDimensions()
        points = self.getTotalNodes()
        ind = [poi*2+dof for poi in range(points) for dof in range(len(self.__vel))]
        self.__velVec = PETSc.Vec().createMPI(
            (( points * len(self.__vel), None)))
        self.__velVec.setValues( ind , np.tile(self.__vel, points) )
        self.__velVec.assemble()

    def regenerateDMPlex(self, dh, dim=1):
        coords, cone , _ = self.generateBody(dh)
        dm = PETSc.DMPlex().createFromCellList(dim, cone,coords)
        return dm

    def saveVTK(self, dir, step=None):
        viewer = PETSc.Viewer()
        if step == None:
            viewer.createVTK('body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        else:
            viewer.createVTK(f"body-{step:05d}", mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.__dom)
        viewer.destroy()

    def setCenter(self, val):
        self.__centerDisplacement = val

    def setElementLong(self, dl, normals):
        self.__normals = normals
        self.__dl = dl

    def computeForce(self, q):
        fx = 0
        fy = 0
        fouris = 0
        points = self.getTotalNodes()
        for poi in range(points):
            nodes = self.__lagNodes[poi]
            fx += q[poi*2] * nodes
            fy += q[poi*2+1] * nodes
            if 16 - nodes < 0:
                print("HAY MAS NODOS!")
            fouris += 16 - nodes
        return fx, fy, fouris

    def getVelocity(self):
        return self.__velVec

    def getElementLong(self):
        return self.__dl

    def getTotalNodes(self):
        return self.lastNode - self.firstNode

    def getLong(self):
        return None

    def setCaracteristicLong(self, val):
        self.__L = val

    def getCaracteristicLong(self):
        return self.__L

    def getCenterBody(self):
        return self.__centerDisplacement

    def getNodeCoordinates(self, node):
        return self.__dom.vecGetClosure(
            self.coordSection, self.coordinates, node + self.firstNode
            ) + self.__centerDisplacement
    
    def getRegion(self):
        return None

    def getDiracs(self, dist):
        dirac = 1
        for r in dist:   
            dirac *= self.dirac(abs(r)/self.__dl)
            dirac /= self.__dl
        return dirac

    def updateBodyParameters(self, t):
        # A1 : f/D = 7.5 & A=D = 1 => f=7.5 & A =1
        velX = 0
        displX = 0
        f = 7.5
        A = 1
        alpha = 2 * pi * self.__Uref / f  
        displY = A * sin(alpha * t)
        velY = A * alpha * cos(alpha * t)
        self.__vel = [velX, velY]
        self.__centerDisplacement = [displX, displY]
        self.updateVelocity()

    def updateVelocity(self):
        points = self.getTotalNodes()
        ind = [poi*2+dof for poi in range(points) for dof in range(len(self.__vel))]
        self.__velVec.setValues( ind , np.tile(self.__vel, points) )
        self.__velVec.assemble()

    def generateBody(self, *args):
        return None, None ,None

class Line(ImmersedBody):
    def generateBody(self, div, **kwargs):
        # this can be improved with lower & upper
        longitud = kwargs['long']
        coords_x = np.linspace(0, longitud, div)
        coords_y = np.array([1]*div)
        coords = np.array([coords_x.T, coords_y.T])
        dl = longitud / (div-1)
        cone= list()
        for i in range(div-1):
            localCone = [i,i+1]
            cone.append(localCone)

        self.generateDMPlex(coords.T, cone)
        self.setCenter(np.array([0,1]))
        self.setCaracteristigLong(longitud)
        self.setElementLong(dl, normals=[1])

    def getLong(self):
        return 1

    def getRegion(self):
        return 1

class Circle(ImmersedBody):
    def __init__(self, vel, center, radius):
        super().__init__(vel, center)
        self.__radius = radius

    def generateBody(self, dh):
        r = self.__radius
        center = self.getCenterBody()
        longTotal = 2*pi*r
        print(dh)
        points = floor(longTotal/dh) + 2
        assert points > 4
        startAng = pi/1000
        angles = np.linspace(0, 2*pi , points)
        x = r * np.cos(angles + startAng) + center[0]
        y = r * np.sin(angles + startAng) + center[1]
        coords = list()
        cone = list()
        for i in range(len(x)-1):
            localCone = [i,i+1]
            coords.append([x[i] , y[i]])
            cone.append(localCone)
        cone[-1][-1] = 0
        dl = sqrt((coords[0][0]-coords[1][0])**2 + (coords[0][1]-coords[1][1])**2)
        # self.generateDMPlex(coords, cone)
        return coords, cone, dl

    def getRadius(self):
        return self.__radius

    def getLong(self):
        return self.__radius*2

    def getRegion(self):
        dl = self.getElementLong()
        return self.__radius + 2*dl

def threeGrid(r):
    """supports only three cell grids"""
    if r <=  0.5:
        return (1 + sqrt(-3*r**2 + 1))/3
    elif r <= 1.5:
        return (5 - 3*r - sqrt(-3*(1-r)**2 + 1))/6
    else:
        return 0

def linear(r):
    """Lineal Dirac discretization"""
    if (r < 1):
        return (1 - r)
    else:
        return 0

def fourGrid(r):
    if r <=  1:
        return (3 - 2*r + sqrt(1 + 4*r - 4*r**2))/8
    elif r <= 2:
        return (5 - 2*r - sqrt(-7+12*r-4*r**2))/8
    else:
        return 0

class EulerNodes:
    def __init__(self, total, dim):
        self.__eulerNodes = list()
        self.__localLagNodes = set()
        self.__totalNodes = total
        self.__dim = dim

    def __repr__(self):
        print(f"Total Nodes in Domain: {self.__totalNodes}")
        print("Nodes affected by Body")
        for eul in self.__eulerNodes:
            print(f"Node Euler: {eul.getNumEuler()} :  {eul.getLagList()}  ")

        print(f"Local Lagrangian num Nodes: {self.__localLagNodes}")
        return "------"

    def getAffectedNodes(self):
        return self.__eulerNodes

    def add(self, eul, lag, diracs):
        euler = EulerNode(eul, lag, diracs)
        self.__eulerNodes.append(euler)
        self.__localLagNodes.update(lag)

    def generate_d_nnz(self):
        d_nnz = [0] * self.__totalNodes * self.__dim
        for eul in self.__eulerNodes:
            nodeNum = eul.getNumEuler()
            for dof in range(self.__dim):
                d_nnz[nodeNum*2+dof] = eul.getNumLag()
        return d_nnz

    def getProblemDimension(self):
        rows = self.__totalNodes * self.__dim
        cols = len(self.__localLagNodes) * self.__dim
        return cols, rows

class EulerNode:
    def __init__(self, num, lag, diracs):
        self.__num = num
        self.__lagNodes = lag
        self.__diracs = diracs

    def __repr__(self):
        return f"Euler Node: {self.__num}"

    def getLagList(self):
        return self.__lagNodes

    def getNumLag(self):
        return len(self.__lagNodes)
    
    def getNumEuler(self):
        return self.__num
        
    def getDiracComputed(self):
        return self.__diracs