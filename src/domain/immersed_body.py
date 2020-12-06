import numpy as np
from math import sin, cos , pi , sqrt, ceil, floor
from petsc4py import PETSc
import logging
import yaml
import logging

class BodiesContainer:
    types = ['circle', 'line', 'box']
    def __init__(self, bodies):
        self.logger = logging.getLogger("Bodies Container")
        self.bodies= list()
        for cfgBody in bodies:
            center = cfgBody['center']
            if cfgBody['type'] == 'circle':
                body = Circle(vel=[0,0], center=center, radius=cfgBody['radius'])
            elif cfgBody['type'] == 'line':
                body = Line(vel=[0,0], center=center)
            elif cfgBody['type'] =='box':
                body = OpenBox(vel=[0,0], center=center)
            else:
                raise Exception("not defined")
            
            if cfgBody['vel'] == 'dynamic':
                body.setIsMoving()
            self.bodies.append(body)

    def createBodies(self, h):
        for i, body in enumerate(self.bodies):
            body.generateDMPlex(h)
            self.logger.info(f"Body number: {i}")
            body.viewState()
        self.locTotalNodes = body.getTotalNodes()
        # exit()

    def getTotalNodes(self):
        nodes = 0
        for body in self.bodies:
            bodyNodes = body.getTotalNodes()
            nodes += bodyNodes
        return nodes

    def getRegion(self):
        radius = 0.5
        distanceCenters = 2
        hCeil = self.getElementLong()
        tot = radius + distanceCenters + hCeil*4
        return tot

    def mapGlobToLocal(self, globNode):
        numBody = 0
        locNode = None
        if globNode >= self.locTotalNodes:
            globNode = globNode - self.locTotalNodes
            numBody +=1
        locNode = globNode
        return locNode, numBody

    def getNodeCoordinates(self, globNode):
        # input is global
        # necesito identificar a que cuerpo pertenece
        numBody = 0
        if globNode >= self.locTotalNodes:
            globNode = globNode - self.locTotalNodes
            numBody +=1
        coord = self.bodies[numBody].getNodeCoordinates(globNode)
        return coord

    def getDiracs(self, dist, h):
        return self.bodies[0].getDiracs(dist, h)

    def getElementLong(self):
        dl = self.bodies[0].getElementLong()
        return dl

    def getCenters(self):
        centers = list()
        for i in self.bodies:
            centers.append(i.getCenterBody())
        return centers

    def setEulerNodes(self, glNode, eulerNodesNum):
        locNode, numBody = self.mapGlobToLocal(glNode)
        self.bodies[numBody].setEulerNodes(locNode, eulerNodesNum)

    def computeForce(self, vec, dt):
        offset = 0
        forces_x = list()
        forces_y = list()
        vec = vec.getArray()
        for body in self.bodies:
            nodes = body.getTotalNodes()
            fx_local, fy_local = body.computeForce(vec[offset*2:(nodes+offset)*2])
            forces_x.append(float(fx_local/dt))
            forces_y.append(float(fy_local/dt))
            offset += nodes
        return forces_x, forces_y

    def setVelRef(self, vel):
        for body in self.bodies:
            body.setVelRef(vel)

    def getVelRef(self, bodyNum=0):
        return self.bodies[bodyNum].getVelRef()

    def updateBodyParameters(self, t):
        for body in self.bodies:
            body.updateBodyParameters(t)

    def viewBodies(self):
        for i, body in enumerate(self.bodies):
            print(f"Body num: {i}")
            # body.view()
            body.viewState()
            # body.viewCoordinates()

    def getVelocity(self):
        # hacer esto para varios cuerpos
        vels = list()
        for body in self.bodies:
        # body = self.bodies[0]
            vel = body.getVelocity()
            vels.append(vel)
        # vel = PETSc.Vec().createNest(vels)
        return PETSc.Vec().createNest(vels)

    def updateVelocity(self):
        for body in self.bodies:
            body.updateVelocity()

    def bodyNumbers(self):
        return len(self.bodies)

class ImmersedBody:
    def __init__(self, vel=[0,0], center=[0,0]):
        self.dirac = fourGrid
        self.__centerDisplacement = center
        self.__startCenter = center
        self.__dl = None
        self.__vel = vel
        self.__Uref = None
        self.logger = logging.getLogger("Body Immersed")
        self.__history = {"times": [], "displ": [], "vel": [] } 
        self.__isStatic = True

    def setVelRef(self, vel):
        self.__Uref = vel

    def getVelRef(self):
        return self.__Uref

    def setIsMoving(self):
        self.__isStatic = False

    def view(self):
        self.logger.info(f"Arc len: {self.__dl} | Dirac Type: {self.dirac.__name__} | Vel Fluid Reference: {self.__Uref} ")

    def viewState(self):
        self.logger.info(f"{self.__class__.__name__} vel: {self.__vel} | Body center position {self.__centerDisplacement}")
        with open('body-history.yaml', 'w') as outfile:
            yaml.dump(self.__history, outfile, default_flow_style=False)
    
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
            viewer.createVTK('body-testing.vtk', mode=PETSc.Viewer.Mode.WRITE)
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
        points = self.getTotalNodes()
        for poi in range(points):
            fx += q[poi*2]
            fy += q[poi*2+1]
        return fx, fy

    def getVelocity(self):
        return self.__velVec

    def setVelocity(self, inds, values ):
        self.__velVec.setValues( inds , values )
        self.__velVec.assemble()

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
    
    def viewCoordinates(self):
        # self.__dom.view()
        print(self.firstNode, self.lastNode)
        a = np.zeros(2)
        for i in range(self.lastNode-self.firstNode):
            coord = self.getNodeCoordinates(i)
            # self.logger.info(f" Node: {i} | Coord {coord}")
            a += coord
        print("final" , a)

    def getRegion(self):
        return None

    # @profile
    def getDiracs(self, dist, h):
        dirac = 1
        for r in dist:   
            dirac *= self.dirac(abs(r)/h)
            dirac /= h
        return dirac

    def updateBodyParameters(self, t):
        # A1 : f/D = 7.5 & A=D = 1 => f=7.5 & A =1
        if self.__isStatic:
            return
        velX = 0 
        displX = 0  + self.__startCenter[0]
        f = 5
        Te = f / self.__Uref
        A = 0.3
        displY = A * sin(2 * pi * t / Te) + self.__startCenter[1]
        velY = 2 * pi * A * cos(2 * pi * t / Te)/Te
        self.__vel = [velX, velY]
        self.__centerDisplacement = [displX, displY]
        self.updateVelocity()
        self.__history["times"].append(t)
        self.__history["displ"].append(self.__centerDisplacement)
        self.__history["vel"].append(self.__vel)

    def updateVelocity(self):
        points = self.getTotalNodes()
        ind = [poi*2+dof for poi in range(points) for dof in range(len(self.__vel))]
        self.__velVec.setValues( ind , np.tile(self.__vel, points) )
        self.__velVec.assemble()

    def generateBody(self, *args):
        return None, None ,None

class Line(ImmersedBody):
    def generateBody(self, dl , longitud=2):
        # this can be improved with lower & upper
        self.__region = longitud
        div = ceil(longitud/dl) 
        coords_x = np.linspace(0, longitud, div)
        coords_y = np.array([0]*div)
        cone= list()
        coords = list()
        for i in range(div-1):
            localCone = [i,i+1]
            cone.append(localCone)
            coords.append([coords_x[i], coords_y[i]])
        coords.append([coords_x[i+1], coords_y[i+1]])
        return coords, cone, dl

    def getLong(self):
        return 1

    def getRegion(self):
        return self.__region

class OpenBox(ImmersedBody):
    def generateBody(self, dl , longitud=1):
        # this can be improved with lower & upper
        cone= list()
        coords = list()
        div = ceil(sqrt(2)/dl)
        x1 , y1 = 0, longitud
        x2 , y2 = -longitud, 0
        x3 , y3 = 0, - longitud
        x4 , y4 = longitud, 0
        x1Tox2 =  np.linspace(x1, x2, div, endpoint=False)
        x2Tox3 = np.linspace(x2, x3, div, endpoint=False)
        x3Tox4 = np.linspace(x3, x4, div, endpoint=False)
        xfinal = np.linspace(x4, x1, div, endpoint=False)

        coords_x = np.append(x1Tox2, [x2Tox3, x3Tox4, xfinal])

        y1Toy2 =  np.linspace(y1, y2, div, endpoint=False)
        y2Toy3 = np.linspace(y2, y3, div, endpoint=False)
        y3Toy4 = np.linspace(y3, y4, div, endpoint=False)
        yfinal = np.linspace(y4, y1, div, endpoint=False)

        coords_y = np.append(y1Toy2, [y2Toy3, y3Toy4, yfinal])

        for i in range(len(coords_x)):
            localCone = [i,i+1]
            cone.append(localCone)
            coords.append([coords_x[i], coords_y[i]])
        cone[-1][-1] = 0

        return coords, cone, dl

    def updateVelocity(self):
        points = self.getTotalNodes()
        velRef = self.getVelRef()
        self.logger.info(f"setting Lid Driven Cavity Velocity: {velRef}")
        velValues = np.zeros(points*2)
        for poi in range(points):
            coord = self.getNodeCoordinates(poi)
            if coord[0] >= 0 and coord[1] >= 0:
                velValues[poi*2] = velRef / sqrt(2)
                velValues[poi*2+1] = - velRef / sqrt(2)
        ind = [poi*2+dof for poi in range(points) for dof in range(2)]
        self.setVelocity(ind, velValues)

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
        longTotal = 2*pi*r
        points =  ceil(longTotal/dh)
        # assert points > 4
        dh = longTotal/points
        startAng = pi/1000
        angles = np.linspace(0, 2*pi , points, endpoint=False)
        x = r * np.cos(angles + startAng)
        y = r * np.sin(angles + startAng)
        coords = list()
        cone = list()
        for i in range(len(x)):
            localCone = [i,i+1]
            coords.append([x[i] , y[i]])
            cone.append(localCone)
        cone[-1][-1] = 0
        dl = 2 * pi * r / len(coords)
        return coords, cone, dl

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

# @profile
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

if __name__ == '__main__':
    logging.basicConfig(level='INFO' )
    bodies = BodiesContainer('side-by-side')
    bodies.createBodies(0.1)
    bodies.viewBodies()