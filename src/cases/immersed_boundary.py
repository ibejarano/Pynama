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
        self.cteValue = [300,0]
        ndiv = 4 * 7
        assert self.dim == 2
        dxMax = self.upper[0] - self.lower[0]
        rawNelem = self.nelem[0]
        nelem =float(rawNelem)
        self.h = dxMax / nelem
        self.body = Body(ndiv)
        self.createIBMMatrix()
        self.saveVTK()
        # self.diracVec = self.mat.K.createVecRight()
        # self.saveDiracVec()

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.solveKLE(startTime, self.vort)
        self.vort = self.getVorticityCorrection()

    # def setDiracVec(self, nodes, d):
    #     self.diracVec.setValues(nodes[::2], d, addv=True)
    #     self.diracVec.setValues(nodes[1::2], d, addv=True)

    # def saveDiracVec(self):
    #     self.diracVec.assemble()
    #     self.diracVec.setName("dirac")
        # self.viewer.saveVec(self.diracVec, timeStep=1)

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        incr = ts.getTimeStep()
        self.vort = self.getVorticityCorrection()
        self.solveKLE(time, self.vort)
        self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveVec(self.diracVec, timeStep=step)
        self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort, self.diracVec])

    def saveVTK(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('dom.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.body.dom)
        viewer.destroy()

    def applyBoundaryConditions(self):
        self.vel.set(0.0)
        velDofs = [nodes*2 + dof for nodes in self.boundaryNodes for dof in range(2)]
        self.vel.setValues(velDofs, np.tile(self.cteValue, len(self.boundaryNodes)))

    def solveKLE(self, time, vort, finalStep=False):
        self.applyBoundaryConditions()
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)
        vorticity_corrected = self.getVorticityCorrection()
        self.solver( self.mat.Rw * vorticity_corrected + self.mat.Krhs * self.vel , self.vel)

    def getVorticityCorrection(self, finalStep=False):
        velocityCorrection = self.computeVelocityCorrection()
        vort = self.mat.Curl.createVecLeft()
        vort.setName("vorticity")
        if not finalStep:
            self.vel += velocityCorrection
        self.mat.Curl(self.vel , vort)
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
        print("Creating ibm Matrices")
        """creates a PETSc Mat type
        """
        dim = self.dim
        nodesLagTot = self.body.getTotalNodes()
        nodesEultotal = len(self.dom.getAllNodes())
        nodeLagWnodesEuler = dict()
        pre_d_nnz_D = list()
        cellsAffected = self.getAffectedCells()
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
        # self.createEmptyVecs()

    def getAffectedCells(self):
        """ONLY VALID FOR CIRCLE BODY"""
        cellStart, cellEnd = self.dom.getHeightStratum(0)
        cells = list()
        circleCenter = np.array([0, 0])
        radius = self.body.getRadius()
        for cell in range(cellStart, cellEnd):
            cellCoords = self.dom.getCellCornersCoords(cell).reshape(( 2 ** self.dim, self.dim))
            cellCentroid = self.computeCentroid(cellCoords)
            dist = cellCentroid - circleCenter
            if dist[0] < (radius + 3*self.body.dl): #Solo tomo una porcion rectangular
                cells.append(cell)
        return cells

    def getEulerNodes(self, cellList, bodyNode):
        points2 = set()
        coordsBodyNode = self.body.getNodeCoordinates(bodyNode)
        for cell in cellList:
            coords = self.dom.getCellCornersCoords(cell).reshape((2**self.dim, self.dim))
            cellCentroid = self.computeCentroid(coords)
            dist = coordsBodyNode - cellCentroid
            if (abs(dist[0]) < self.body.dl*2) & (abs(dist[1]) < self.body.dl*3):
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
        # if nodeBody == 0:
        #     self.setDiracVec(pointsClosest, distanceClosest)
        #     print(f"{nodeBody =}")
        #     print(f"{nodesClose =}")
        #     print(f"{pointsClosest =}")
        #     print(f"{distanceClosest =}")
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
        vort = self.mat.Curl.createVecLeft()
        vort.setName("vorticity")
        self.vel += self.computeVelocityCorrection()
        self.mat.Curl(self.vel , vort)
        return vort        

class Body:
    def __init__(self, divisions):
        """Immersed boundary method class
        Arguments:
            domLag {string} -- it does indicates the location of .msh Gmsh File
        """

        # FIXME: i tried to create from gmsh but i cant
        # self.domLag = PETSc.DMPlex().createFromFile(domLag)
        self.radius = 1
        self.bodyMovement = False  # FIXME dtype can be a PETSc Vec in a future
        # FIXME: getHeightStratum(1) is only valid for dim=2
        self.dirac = linear
        self.dom , self.dl = generateCircleDMPlex(self.radius,divisions)
        self.setUpDimensions()

    def setUpDimensions(self):
        # self.getInternalBodyNodes()
        self.firstNode, self.lastNode = self.dom.getHeightStratum(1)
        self.coordinates = self.dom.getCoordinatesLocal()
        self.coordSection = self.dom.getCoordinateSection()
        self._centerDisplacement = np.array([0,0])

    def createVtkFile(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('ibCircle.vtk')
        viewer.view(self.dom)
        viewer.destroy()

    def getRadius(self):
        return self.radius

    def getElementLong(self):
        return self.dl

    def getTotalNodes(self):
        return self.lastNode - self.firstNode

    def getNodeCoordinates(self, node):
        """Gets a List with all the coordinates from a node
        Arguments:
            node {int} -- Node from Lagrangian mesh
        Raises:
            Exception -- if node number exceedes the space range
        Returns:
            [float] -- x , y [,z] coordinates
        """
        if node + self.firstNode >= self.lastNode:
            raise Exception('node parameter must be in local numbering!')
        return self.dom.vecGetClosure(
            self.coordSection, self.coordinates, node + self.firstNode
            ) + self._centerDisplacement

    # BODY MOVEMENTS FUNCTIONS

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
            self._velocity.setValues(velIndex, vel, False)
        # print('Body vel size', self._BodyVelocity.getSize())
        self._velocity.assemble()

    def updateCoordinates(self, t):
        displX = 0
        displY = t * sin(t/2)
        self._centerDisplacement = np.array([displX , displY])

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

def generateCircleDMPlex(radius,n):
    r= radius
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
    return PETSc.DMPlex().createFromCellList(1, cone,coords) ,  dl