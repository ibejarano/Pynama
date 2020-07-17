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
import matplotlib.pyplot as plt

# ibm imports
from math import sqrt, sin, pi, ceil

class ImmersedBoundaryStatic(FreeSlip):
    def setUp(self):
        super().setUp()
        self.boundaryNodes = self.getBoundaryNodes()
        self.createIBMMatrix()
        self.body.saveVTK()

    def readBoundaryCondition(self, inputData):
        try:
            re = inputData['constant']['re']
            L = self.body.getCaracteristicLong()
            vel_x = re*(self.mu/self.rho) / L
            self.U_ref = vel_x
            self.cteValue = [vel_x,0]
        except:
            vel = inputData['constant']['vel']
            self.U_ref = (vel[0]**2 + vel[1]**2)**0.5
            self.cteValue = [vel_x,0]

    def readDomainData(self, inputData):
        super().readDomainData(inputData)
        numElements = self.nelem[0]
        self.h = (self.upper[0] - self.lower[0])/numElements
        if self.ngl == 3:
            self.h /= 2
        elif self.ngl > 3:
            raise Exception("High Order nodes not implemented")
        self.body = self.createBody(inputData['body'])

    def startSolver(self):
        self.computeInitialCondition(startTime= 0.0)
        self.vort.set(0.0)
        self.computeVelocityCorrection(NF=4)
        self.operator.Curl.mult(self.vel, self.vort)
        self.ts.setSolution(self.vort)
        cds = list()
        clifts = list()
        times = list()
        for i in range(100):
            self.ts.step()
            step = self.ts.getStepNumber()
            time = self.ts.time
            dt = self.ts.getTimeStep()
            qx , qy = self.computeVelocityCorrection(NF=4)
            cd, cl = self.computeDragForce(qx / dt, qy / dt)
            cds.append(cd)
            clifts.append(cl)
            times.append(time)
            self.operator.Curl.mult(self.vel, self.vort)
            self.viewer.saveVec(self.vel, timeStep=step)
            self.viewer.saveVec(self.vort, timeStep=step)
            self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort])
            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Cd {cd} | Cl {cl} ")
            # print(f"printing info {self.ts.getTime()= } {self.ts.getSolveTime()= } {self.ts.getPrevTime() = }")
            # print(f"printing info {self.ts.getTimeStep()= } {self.ts.getStepNumber()= }")
            self.ts.setSolution(self.vort)
            self.viewer.writeXmf("ibm-static")
        # print(times)
        # print(cds)
        # print(clifts)
        plt.figure(figsize=(10,10))
        plt.plot(times, cds, 'r-', label='c_drag')
        plt.plot(times, clifts, 'b-', label='c_lift')
        plt.legend()
        plt.xlabel("times [s]")
        plt.ylabel("C_d")
        plt.grid(True)
        # plt.show()
        nombre = "pruebitas"
        plt.savefig(f"cdsclsVtime-{nombre}.png")

    def computeDragForce(self, fd, fl):
        U = self.U_ref
        denom = 0.5 * self.rho * (U**2)
        cd = fd/denom
        return cd, fl/denom

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.solveKLE(startTime, self.vort)

    def setUpSolver(self):
        self.solver = KspSolver()
        self.solver.createSolver(self.mat.K, self.comm)
        self.vel = self.mat.K.createVecRight()
        self.vel.setName("velocity")
        self.vort = self.mat.Rw.createVecRight()
        self.vort.setName("vorticity")
        self.vort.set(0.0)

        self.vel_correction = self.vel.copy()
        self.vel_correction.setName("velocity_correction")
        self.vort_correction = self.vort.copy()
        self.vort_correction.setName("vorticity_correction")

        self.virtualFlux = self.S.createVecRight()
        self.virtualFlux.setName("virtual_flux")
        self.ibm_rhs = self.S.createVecRight()

        sK, eK = self.mat.K.getOwnershipRange()
        locRowsK = eK - sK

        self._VtensV = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)
        self._Aux1 = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)

    def applyBoundaryConditions(self, a, b):
        self.vel.set(0.0)
        velDofs = [nodes*2 + dof for nodes in self.boundaryNodes for dof in range(2)]
        self.vel.setValues(velDofs, np.tile(self.cteValue, len(self.boundaryNodes)))

    def computeVelocityCorrection(self, NF=1):
        fx = 0
        fy = 0
        bodyVel = self.body.getVelocity()
        for i in range(NF):
            self.H.mult(self.vel, self.ibm_rhs)
            self.ksp.solve(bodyVel - self.ibm_rhs, self.virtualFlux)
            self.S.mult(self.virtualFlux, self.vel_correction)
            aux = self.virtualFlux.getArray()
            fx_part, fy_part = self.body.computeForce(aux)
            fx += fx_part
            fy += fy_part
            self.vel += self.vel_correction
        return abs(fx*self.rho), abs(fy*self.rho)

    def createBody(self, inp):
        vel = inp['vel']
        body = inp['type']
        if body['name'] == "circle":
            radius = body['radius']
            center = body['center']
            ibmBody = Circle(vel)
            ibmBody.generateBody(self.h, radius=radius)
            return ibmBody

    def createIBMMatrix(self):
        rows = self.body.getTotalNodes() * self.dim
        cols = len(self.dom.getAllNodes()) * self.dim
        bodyRegion = self.body.getRegion()
        cellsAffected = self.getAffectedCells(bodyRegion)
        nodes = self.dom.getGlobalNodesFromEntities(cellsAffected, shared=False)
        d_nnz_D = len(nodes)
        o_nnz_D = 0

        self.H = PETSc.Mat().createAIJ(size=(rows, cols), 
            nnz=(d_nnz_D, o_nnz_D), 
            comm=self.comm)
        self.H.setUp()

        lagNodes = self.body.getTotalNodes()
        eulerIndices = [node*self.dim+dof for node in nodes for dof in range(self.dim)]
        eulerCoords = self.dom.getNodesCoordinates(nodes)
        for lagNode in range(lagNodes):
            dirac = self.computeDirac(lagNode, eulerCoords)
            for dof in range(self.dim):
                self.H.setValues(lagNode*self.dim+dof, eulerIndices[dof::self.dim], dirac)

        self.H.assemble()
        self.S = self.H.copy().transpose()

        dl = self.body.getElementLong()
        self.S.scale(dl)
        self.H.scale(self.h**2)
        A = self.H.matMult(self.S)
        self.ksp = KspSolver()
        self.ksp.createSolver(A, self.comm)

    def getAffectedCells(self, xSide, ySide=None , center=[0,0]):
        try:
            assert ySide
        except:
            ySide = xSide

        cellStart, cellEnd = self.dom.getHeightStratum(0)
        cells = list()
        for cell in range(cellStart, cellEnd):
            cellCoords = self.dom.getCellCornersCoords(cell).reshape(( 2 ** self.dim, self.dim))
            cellCentroid = self.computeCentroid(cellCoords)
            dist = cellCentroid - center
            if abs(dist[0]) < (xSide) and abs(dist[1]) < (ySide):
                cells.append(cell)
        return cells

    def computeDirac(self, lagPoint, eulerCoords):
        diracs = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            d = self.body.getDiracs(dist)
            diracs.append(d)
        return diracs

    @staticmethod
    def computeCentroid(corners):
        return np.mean(corners, axis=0)



class ImmersedBoundaryDynamic(ImmersedBoundaryStatic):
    def getVorticityCorrection(self, t, finalStep=False):
        """Main function to be called after a Converged Time Step"""
        vort = self.operator.Curl.createVecLeft()
        return vort 

class ImmersedBody:
    def __init__(self, vel=[0,0]):
        self.dirac = fourGrid
        self.__centerDisplacement = [0,0]
        self.__dl = None
        self.__vel = vel
    
    def setUpDimensions(self):
        self.firstNode, self.lastNode = self.__dom.getHeightStratum(1)
        self.coordinates = self.__dom.getCoordinatesLocal()
        self.coordSection = self.__dom.getCoordinateSection()

    def generateDMPlex(self, coords, cone, dim=1):
        self.__dom = PETSc.DMPlex().createFromCellList(dim, cone,coords)
        self.setUpDimensions()
        points = self.getTotalNodes()
        ind = [poi*2+dof for poi in range(points) for dof in range(len(self.__vel))]
        self.__velVec = PETSc.Vec().createMPI(
            (( points * len(self.__vel), None)))
        self.__velVec.setValues( ind , np.tile(self.__vel, points) )
        self.__velVec.assemble()

    def saveVTK(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.__dom)
        viewer.destroy()

    def setCenter(self, val):
        self.__centerDisplacement = val

    def setElementLong(self, dl, normals):
        self.__normals = normals
        self.__dl = dl

    def computeForce(self, q):
        # fx = q[0]*norm[0] + q[2]*norm[2]
        # fy = q[1]*norm[1] + q[3]*norm[3]
        fx = 0
        fy = 0
        points = self.getTotalNodes()
        for poi in range(points):
            coord = self.getNodeCoordinates(poi)
            d_x = coord[0] * self.__dl / 0.5
            d_y = coord[1] * self.__dl / 0.5
            fx += q[poi*2]*d_x
            fy += q[poi*2+1]*d_y
        return fx, fy

    def getVelocity(self):
        return self.__velVec

    def getElementLong(self):
        return self.__dl

    def getTotalNodes(self):
        return self.lastNode - self.firstNode

    def setCaracteristigLong(self, val):
        self.__L = val

    def getCaracteristicLong(self):
        return self.__L

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
        self.setCenter(np.array([0,0]))
        self.setCaracteristigLong(longitud)
        self.setElementLong(dl, normals=[1])

    def getRegion(self):
        return 1

class Circle(ImmersedBody):
    def generateBody(self, dh, **kwargs):
        r = kwargs['radius']
        rev = 2*pi
        n = ceil(rev*r/dh)
        assert n > 4
        div = rev/n
        startAng = 0.1*pi
        # startAng = 0
        angles = np.arange(0, rev+div , div)
        x = r * np.cos(angles + startAng)
        y = r * np.sin(angles + startAng)
        coords = list()
        cone = list()
        norms = list()
        for i in range(len(x)-1):
            localCone = [i,i+1]
            coords.append([x[i] , y[i]])
            norms.append([x[i]/r , y[i]/r])
            cone.append(localCone)
        cone[-1][-1] = 0
        dl = sqrt((coords[0][0]-coords[1][0])**2 + (coords[0][1]-coords[1][1])**2)

        self.setCenter(np.array([0,0]))
        self.setElementLong(dl, normals=norms)
        self.setCaracteristigLong(r*2)
        self.radius = r
        self.generateDMPlex(coords, cone)

    def getRadius(self):
        return self.radius

    def getRegion(self):
        dl = self.getElementLong()
        return self.radius + dl

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