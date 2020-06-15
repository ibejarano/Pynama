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
        self.setUpGeneral()
        self.cteValue = [1,0]

        radius = 0.5
        ndiv = 16
        assert self.dim == 2
        dxMax = self.upper[0] - self.lower[0]
        rawNelem = self.nelem[0]
        nelem =float(rawNelem)
        self.h = dxMax / nelem

        self.setUpBoundaryConditions()
        self.body = Body(radius, ndiv)
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()
        self.createIBMMatrix()


    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)

    def applyBoundaryConditions(self, time, bcNodes):
        self.vel.set(0.0)
        self.vel = self.dom.applyValuesToVec(bcNodes, self.cteValue, self.vel)

    def solveKLE(self, time, vort, finalStep=False):
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        # compute velocity prediction
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)
        # use the velocity to get vorticity correction
        vorticity_corrected = self.getVorticityCorrection(time, finalStep=finalStep)
        # compute velocity again
        self.solver( self.mat.Rw * (vorticity_corrected+vort) + self.mat.Krhs * self.vel , self.vel)

    def getVorticityCorrection(self, t,finalStep=False):
        """Main function to be called after a Converged Time Step"""
        # TODO: body movement algos in a new subclass
        # if self.bodyMovement:
        #     self.D.destroy()
        #     self.Dds.destroy()
        #     self.ksp.destroy()
        #     self.computeBodyVelocity(t)
        #     self.updateBodyCoordinates(t)
        #     self.buildMatrices()
        #     print('Center Displacement' ,self._centerDisplacement)
        #     print('Body Velocity' ,self._BodyVelocity.getArray())
        #     # tomar tiempo -> computar vel # DONE
        #     # vel * t -> Vec desplazamiento # DONE
        #     # Computar nueva matriz D 
        velocityCorrection = self.getVelocityCorrection()
        vort = self.mat.Curl.createVecLeft()
        if finalStep:
            self.vel += velocityCorrection
        self.mat.Curl(self.vel +  velocityCorrection , vort)
        return vort

    def getVelocityCorrection(self):
        velCorrection = self.computeVelocityCorrection(self.vel)
        return velCorrection

    def computeVelocityCorrection(self, predictedVel):
        predictedVel = self.vel
        velCorrection = self.vel.duplicate()
        rhs = self.D.createVecLeft()
        virtualFlux = self.Dds.createVecRight()
        self.D.mult(-predictedVel*(self.h**2) , rhs)
        self.ksp.solve(rhs, virtualFlux)
        self.Dds.mult(virtualFlux, velCorrection)
        return velCorrection

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        # vort = ts.getSolution()
        self.logger.info(f"Converged: Step {step} Time {time}")
        # lo de abajo en otro lado
        # self.logger.info(f"Reason: {ts.reason}")
        # self.logger.info(f"max vel: {self.vel.max()}")
        self.solveKLE(time,self.vort, finalStep=True)
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort])

    # Creating matrix
    def createIBMMatrix(self):
        print("Creating ibm Matrices")
        """creates a PETSc Mat type
        """
        dim = self.dim
        nodesLagTot = self.body.getTotalNodes()
        nodesEultotal = len(self.dom.getAllNodes())
        #d_nnz_Dds = [nodesLagTot*dim] * nodesEultotal * dim
        #o_nnz_Dds = [0] * nodesEultotal * dim
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
            if dist[0] < (radius + 3*self.h): #Solo tomo una porcion rectangular
                cells.append(cell)
        return cells

    def getEulerNodes(self, cellList, bodyNode):
        points2 = set()
        coordsBodyNode = self.body.getNodeCoordinates(bodyNode)
        for cell in cellList:
            coords = self.dom.getCellCornersCoords(cell).reshape((2**self.dim, self.dim))
            cellCentroid = self.computeCentroid(coords)
            dist = coordsBodyNode - cellCentroid
            if (abs(dist[0]) < self.h*2.5) & (abs(dist[1]) < self.h*2.5):
                listaux = self.dom.getGlobalNodesFromCell(cell, shared=False)
                points2.update(listaux)
        return points2

    def getClosestDistance(self, nodeBody, nodes):
        nodes = list(nodes)
        distanceClosest = list()
        pointsClosest = list()
        coordBodyNode = self.body.getNodeCoordinates(nodeBody)
        nodeCoordinates = self.dom.getNodesCoordinates(nodes)
        for node_ind, coords in enumerate(nodeCoordinates):
            dist = coords - coordBodyNode
            d = 1
            for x in dist:
                d *= self.body.dirac(x/self.h)
            d /= (self.h**2)
            distanceClosest.append(d)
            pointsClosest.extend(self.dom.getVelocityIndex([nodes[node_ind]]))
        return distanceClosest, pointsClosest

    @staticmethod
    def computeCentroid(corners):
        return np.mean(corners, axis=0)

class Body:
    def __init__(self, radius, divisions):
        """Immersed boundary method class
        Arguments:
            domLag {string} -- it does indicates the location of .msh Gmsh File
        """

        # FIXME: i tried to create from gmsh but i cant
        # self.domLag = PETSc.DMPlex().createFromFile(domLag)

        self.bodyMovement = False  # FIXME dtype can be a PETSc Vec in a future
        # FIXME: getHeightStratum(1) is only valid for dim=2
        self.dirac = threeGrid
        self.dom , self.dl = generateCircleDMPlex(radius,divisions)
        self.radius = radius
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

    def buildMatrices(self):
        self.createEmptyVecs()

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

    def createEmptyVecs(self):
        print("Creating empty vecs")
        self._VelCorrection = self.D.createVecRight()
        self._BodyVelocity = self.Dds.createVecRight()
        self._BodyVelocity.set(0.0)
        self.rhs = self.D.createVecLeft()
        self.virtualFlux = self.Dds.createVecRight()

    def showMatInfo(self, mat):
        print("Matriz size: ", mat.sizes)
        print("Matriz information: ", mat.getInfo())
   
    def printVec(self,vec):
        arr = vec.getArray()
        print("x comp: ", arr[::2])
        print("y comp: ", arr[1::2])
        print("")


    # BODY MOVEMENTS FUNCTIONS

    def computeBodyVelocity(self, t):
        velX = 0
        velY = sin(t/2)
        # print('Computed Vel Y' , velY)
        nodes = self.lastNode - self.firstNode
        for i in range(nodes):
            velIndex = [i*2 , i*2 +1] 
            # print('velocity ind', velIndex)
            vel =  [velX, velY]
            # print('Vel', vel)
            self._BodyVelocity.setValues(velIndex, vel, False)
        # print('Body vel size', self._BodyVelocity.getSize())
        self._BodyVelocity.assemble()

    def updateBodyCoordinates(self, t):
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