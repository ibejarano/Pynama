import sys
import petsc4py
petsc4py.init(sys.argv)
from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer
from solver.ksp_solver import KspSolver

# ibm imports
from math import sqrt, sin, pi

class ImmersedBoundaryStatic(FreeSlip):
    def setUp(self):
        super().setUp()

        self.boundaryNodes = self.getBoundaryNodes()
        self.cteValue = [10,0]
        ndiv = 4 * 1
        assert self.dim == 2
        dxMax = self.upper[0] - self.lower[0]
        rawNelem = self.nelem[0]
        nelem =float(rawNelem)
        self.h = dxMax / nelem
        self.body = Circle()
        self.body.generateBody(ndiv, radius=0.5)
        # self.createIBMMatrix()
        self.body.saveVTK()

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.solveKLE(startTime, self.vort)
        self.vort = self.getVorticityCorrection()

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        incr = ts.getTimeStep()
        self.vort = self.getVorticityCorrection()
        # self.solveKLE(time, self.vort)
        self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort])

    def applyBoundaryConditions(self):
        self.vel.set(0.0)
        velDofs = [nodes*2 + dof for nodes in self.boundaryNodes for dof in range(2)]
        self.vel.setValues(velDofs, np.tile(self.cteValue, len(self.boundaryNodes)))

    def solveKLE(self, time, vort, finalStep=False):
        self.applyBoundaryConditions()
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)
        vorticity_corrected = self.getVorticityCorrection()
        self.solver( self.mat.Rw * (vorticity_corrected+vort) + self.mat.Krhs * self.vel , self.vel)

    def getVorticityCorrection(self, finalStep=False):
        velocityCorrection = self.computeVelocityCorrection()
        vort = self.operator.Curl.createVecLeft()
        vort.setName("vorticity")
        if not finalStep:
            self.vel += velocityCorrection
        self.operator.Curl(velocityCorrection , vort)
        return vort

    def computeVelocityCorrection(self):
        velCorrection = self.vel.copy()
        virtualFlux = self.Dds.createVecRight()
        rhs = self.Dds.createVecRight()
        self.D.mult(self.vel * (self.h**2), rhs)
        self.ksp.solve( -rhs, virtualFlux)
        self.Dds.mult(virtualFlux, velCorrection)
        return velCorrection

    def createIBMMatrix(self):
        dim = self.dim
        nodesLagTot = self.body.getTotalNodes()
        nodesEultotal = len(self.dom.getAllNodes())
        nodeLagWnodesEuler = dict()
        pre_d_nnz_D = list()
        cellsAffected = self.getAffectedCells(None)
        for nodeLag in range(nodesLagTot):
            nodesEuler = self.getEulerNodes(cellsAffected, nodeLag)
            nodeLagWnodesEuler.update({nodeLag: nodesEuler})
            pre_d_nnz_D.append(len(nodesEuler))

        d_nnz_D = [d for d in pre_d_nnz_D for j in range(dim)]
        o_nnz_D = [0] * nodesLagTot * dim

        self.D = PETSc.Mat().createAIJ(
            size=(nodesLagTot*dim, nodesEultotal*dim), 
            nnz=(d_nnz_D, o_nnz_D), 
            comm=self.comm
            )

        self.D.setUp()
        for nodeBody, nodes in nodeLagWnodesEuler.items():
            dist, eulerClosest = self.getClosestDistance(
                nodeBody, nodes)
            self.D.setValues(
                nodeBody*2, eulerClosest[::2], dist
                )
            self.D.setValues(
                nodeBody*2+1, eulerClosest[1::2], dist
                )
        
        self.D.assemble()
        self.Dds = self.D.copy().transpose()

        dl = self.body.getElementLong()
        self.Dds.scale(dl)
        A = self.D.matMult(self.Dds)
        A.scale(self.h**2)
        self.ksp = KspSolver()
        self.ksp.createSolver(A, self.comm)

    def getAffectedCells(self, xSide, ySide=None , center=None):
        try:
            assert ySide
        except:
            ySide = xSide

        cellStart, cellEnd = self.dom.getHeightStratum(0)

        cells = list()

        try:
            assert center[0]
        except:
            center = np.array([0, 0])
        for cell in range(cellStart, cellEnd):
            cellCoords = self.dom.getCellCornersCoords(cell).reshape(( 2 ** self.dim, self.dim))
            cellCentroid = self.computeCentroid(cellCoords)
            dist = cellCentroid - center
            if abs(dist[0]) < (xSide) and abs(dist[1]) < (ySide):
                cells.append(cell)
        return cells

    def getEulerNodes(self, cellList, bodyNode):
        points2 = set()
        coordsBodyNode = self.body.getNodeCoordinates(bodyNode)
        dl = self.body.getElementLong()
        for cell in cellList:
            coords = self.dom.getCellCornersCoords(cell).reshape((2**self.dim, self.dim))
            cellCentroid = self.computeCentroid(coords)
            dist = coordsBodyNode - cellCentroid
            if (abs(dist[0]) < dl*2) & (abs(dist[1]) < dl*2):
                listaux = self.dom.getGlobalNodesFromCell(cell, shared=False)
                points2.update(listaux)
        return points2

    def getClosestDistance(self, nodeBody, nodes):
        nodes = list(nodes)
        distanceClosest = list()
        pointsClosest = list()
        nodesClose = list()
        coordBodyNode = self.body.getNodeCoordinates(nodeBody)
        nodeCoordinates = self.dom.getNodesCoordinates(nodes)
        for node_ind, coords in enumerate(nodeCoordinates):
            dist = coords - coordBodyNode
            d = 1
            for x in dist:
                d *= self.body.dirac(x/self.h)
            d /= (self.h**2)
            if d > 0:
                distanceClosest.append(d)
                nodesClose.append(nodes[node_ind])
                pointsClosest.extend(self.dom.getVelocityIndex([nodes[node_ind]]))
        return distanceClosest, pointsClosest

    @staticmethod
    def computeCentroid(corners):
        return np.mean(corners, axis=0)

class ImmersedBoundaryDynamic(ImmersedBoundaryStatic):
    def getVorticityCorrection(self, t, finalStep=False):
        """Main function to be called after a Converged Time Step"""
        self.D.destroy()
        self.Dds.destroy()
        self.ksp.destroy()
        self.body.computeVelocity(t)
        self.body.updateCoordinates(t)
        self.buildMatrices()
        vort = self.operator.Curl.createVecLeft()
        vort.setName("vorticity")
        self.vel += self.computeVelocityCorrection()
        self.operator.Curl(self.vel , vort)
        return vort        

class ImmersedBody:
    def __init__(self):
        self.dirac = threeGrid
        self.__centerDisplacement = [0,0]
        self.__dl = None
    
    def setUpDimensions(self):
        self.firstNode, self.lastNode = self.__dom.getHeightStratum(1)
        self.coordinates = self.__dom.getCoordinatesLocal()
        self.coordSection = self.__dom.getCoordinateSection()

    def generateDMPlex(self, coords, cone, dim=1):
        self.__dom = PETSc.DMPlex().createFromCellList(dim, cone,coords)
        self.setUpDimensions()

    def saveVTK(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.__dom)
        viewer.destroy()

    def setCenter(self, val):
        self.__centerDisplacement = val

    def setElementLong(self, dl):
        self.__dl = dl

    def getElementLong(self):
        return self.__dl

    def getTotalNodes(self):
        return self.lastNode - self.firstNode

    def getNodeCoordinates(self, node):
        return self.__dom.vecGetClosure(
            self.coordSection, self.coordinates, node + self.firstNode
            ) + self.__centerDisplacement
    
    def getRegion(self):
        return None

    def computeDirac(self, eulerCoord):
        points = list()
        computedDiracs = list()
        allPoints = self.getTotalNodes()
        for poi in range(allPoints):
            coord = self.getNodeCoordinates(poi)
            dist = coord - eulerCoord
            dirac = 1
            for d in dist:
                dirac *= self.dirac(d/self.__dl)
            if dirac > 0:
                computedDiracs.append(dirac)
                points.append(poi)
        return points, computedDiracs

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
        self.__dl = dl
        self.__centerDisplacement = [0,1]
        self.generateDMPlex(coords.T, cone)

    def getRegion(self):
        return 1

class Circle(ImmersedBody):
    def generateBody(self, n, **kwargs):
        r = kwargs['radius']
        rev = 2 * pi
        div = rev/n
        angles = np.arange(0, rev+div , div)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        coords = list()
        cone = list()
        for i in range(len(x)-1):
            localCone = [i,i+1]
            coords.append([x[i] , y[i]])
            cone.append(localCone)
        cone[-1][-1] = 0
        dl = sqrt((coords[0][0]-coords[1][0])**2 + (coords[0][1]-coords[1][1])**2)

        self.setCenter(np.array([0,0]))
        self.setElementLong(dl)
        self.radius = r
        self.generateDMPlex(coords, cone)

    def getRadius(self):
        return self.radius

    def getRegion(self):
        return self.radius

    def computeVelocity(self, t):
        velX = 0
        velY = sin(t/2)
        # print('Computed Vel Y' , velY)
        nodes = self.lastNode - self.firstNode
        for i in range(nodes):
            velIndex = [i*2 , i*2 +1] 
            # print('velocity ind', velIndex)
            vel =  [velX, velY]
            # print('Vel', vel)
            self.__velocity.setValues(velIndex, vel, False)
        # print('Body vel size', self._BodyVelocity.getSize())
        self.__velocity.assemble()

    def updateCoordinates(self, t):
        displX = 0
        displY = t * sin(t/2)
        self.__centerDisplacement = np.array([displX , displY])

def threeGrid(r):
    """supports only three cell grids"""
    accum = 1
    r = abs(r)
    if r <=  0.5:
        accum *= (1 + sqrt(-3*r**2 + 1))/3
    elif r <= 1.5:
        accum *= (5 - 3*r - sqrt(-3*(1-r)**2 + 1))/6
    else:
        return 0
    return accum

def linear(r):
    """Lineal Dirac discretization"""
    accum = 1
    r = abs(r)
    if (r < 1):
        accum *= (1 - r)
    else:
        return 0
    return accum

def fourGrid(r):
    accum = 1
    r = abs(r)
    if r <=  1:
        accum *= (3 - 2*r + sqrt(1 + 4*r - 4*r**2))/8
    elif r <= 2:
        accum *= (5 - 2*r - sqrt(-7+12*r-4*r**2))/8
    else:
        return 0
    return accum

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