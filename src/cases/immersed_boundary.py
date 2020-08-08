import sys
import petsc4py
petsc4py.init(sys.argv)
from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer
from viewer.plotter import DualAxesPlotter
from solver.ksp_solver import KspSolver
from domain.immersed_body import Circle, Line, BodiesContainer
import matplotlib.pyplot as plt
import yaml
# ibm imports
from math import sqrt, sin, pi, ceil

class ImmersedBoundaryStatic(FreeSlip):
    def setUp(self):
        super().setUp()
        self.boundaryNodes = self.getBoundaryNodes()
        cells = self.getAffectedCells(6)
        self.collectedNodes, self.maxNodesPerLag = self.collectNodes(cells)
        self.createEmptyIBMMatrix()
        self.buildIBMMatrix()

        name1= r'Coef. de arrastre $C_D$'
        name2= r'Coef. de empuje $C_{L}$'
        self.plotter = DualAxesPlotter(name1, name2)

    def readBoundaryCondition(self, bc):
        # print(bc)
        try:
            re = bc['constant']['re']
            L = 1
            vel_x = re*(self.mu/self.rho) / L
            self.U_ref = vel_x
            self.cteValue = [vel_x,0]
            self.re = re
        except:
            vel = bc['constant']['vel']
            self.U_ref = (vel[0]**2 + vel[1]**2)**0.5
            self.cteValue = [vel_x,0]

    def setUpDomain(self):
        super().setUpDomain()
        # bodies = self.config.get("body")
        self.h = 0.03636
        # self.h = (20/50)
        # for body in bodies:
            # self.body = self.createBody(body)
        self.body = BodiesContainer('side-by-side')
        self.body.createBodies(self.h)

    def buildOperators(self):
        # cornerCoords = self.dom.getCellCornersCoords(cell=0)
        # localOperators = self.elemType.getElemKLEOperators(cornerCoords)
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            cornerCoords = self.dom.getCellCornersCoords(cell)
            localOperators = self.elemType.getElemKLEOperators(cornerCoords)
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            self.operator.setValues(localOperators, nodes)
        # self.operator.weigDivSrT.assemble()
        # self.weigArea = self.operator.weigDivSrT.copy()
        self.operator.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"Operators Matrices builded - IBM")

    def startSolver(self):
        self.computeInitialCondition(startTime= 0.0)
        self.ts.setSolution(self.vort)
        cds = list()
        clifts = list()
        times = list()
        maxSteps = self.ts.getMaxSteps()
        self.body.viewBodies()
        for i in range(maxSteps):
            self.ts.step()
            step = self.ts.getStepNumber()
            time = self.ts.time
            dt = self.ts.getTimeStep()
            qx , qy, fs = self.computeVelocityCorrection(NF=1)
            cd, cl = self.computeDragForce(qx / dt, qy / dt)
            cds.append(cd)
            clifts.append(cl)
            times.append(time)
            self.operator.Curl.mult(self.vel, self.vort)
            self.ts.setSolution(self.vort)
            self.logger.info(f"Nodos Off {fs}  Converged: Step {step:4} | Time {time:.4e} | Cd {cd:.6f} | Cl {cl:.3f}")
            if time > self.ts.getMaxTime():
                break
            elif i % 10 == 0:
                self.viewer.saveData(step, time, self.vort, self.vel)
                self.viewer.writeXmf(self.caseName)

        self.plotter.updatePlot(times, cds, clifts, realTimePlot=False)
        data = {"times": times, "cd": cds, "cl": clifts}
        runName = f"{self.caseName}-{self.re}"
        with open(runName+'.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        self.plotter.savePlot(runName)

    def computeDragForce(self, fd, fl):
        U = self.U_ref
        denom = 0.5 * (U**2)
        cd = fd/denom
        return float(cd), float(fl/denom)

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.solveKLE(startTime, self.vort)
        self.computeVelocityCorrection(NF=1)
        self.operator.Curl.mult(self.vel, self.vort)

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
        self.vort.setValues( self.boundaryNodes, np.zeros(len(self.boundaryNodes)) , addv=False )

    def computeVelocityCorrection(self, NF=1):
        fx = 0
        fy = 0
        fs = 0
        aux = self.vel.copy()
        # bodyVel = self.body.getVelocity()
        for i in range(NF):
            self.H.mult(self.vel, self.ibm_rhs)
            self.ksp.solve( - self.ibm_rhs, self.virtualFlux)
            self.S.mult(self.virtualFlux, self.vel_correction)
            # fx += fx_part
            # fy += fy_part
            self.vel += self.vel_correction
            aux = self.virtualFlux
            # self.H.multTranspose(self.ibm_rhs, aux)
            # fx_part, fy_part, fs = self.body.computeForce(aux)
            # fx += fx_part
            # fy += fy_part
        return -fx*self.h**2,  -fy*self.h**2, fs

    def createBody(self, body):
        vel = body['vel']
        geo = body['type']
        if geo == "circle":
            radius = body['radius']
            center = body['center']
            ibmBody = Circle(vel, center, radius)
            ibmBody.generateDMPlex(self.h)
            return ibmBody

    # @profile
    def createEmptyIBMMatrix(self):
        rows = self.body.getTotalNodes() * self.dim
        cols = len(self.dom.getAllNodes()) * self.dim
        d_nnz_D = self.maxNodesPerLag
        o_nnz_D = 0

        self.H = PETSc.Mat().createAIJ(size=(rows, cols), 
            nnz=(d_nnz_D, o_nnz_D), 
            comm=self.comm)
        self.H.setUp()

    # @profile
    def buildIBMMatrix(self):
        nodes = set()
        for lagNode in self.collectedNodes.keys():
            data = self.collectedNodes[lagNode]
            coords = data['coords']
            eulerNodes = data['nodes']
            eulerIndices = [node*self.dim+dof for node in eulerNodes for dof in range(self.dim)]
            dirac = self.computeDirac(lagNode, coords)
            for dof in range(self.dim):
                self.H.setValues(lagNode*self.dim+dof, eulerIndices[dof::self.dim], dirac)

            nodes |= set(eulerNodes[np.array(dirac) > 0])

        self.H.assemble()
        self.S = self.H.copy().transpose()
        dl = self.body.getElementLong()
        self.S.scale(dl)
        # self.H.diagonalScale(R=self.weigArea)
        self.H.scale(self.h**2)
        # self.weigArea.destroy()
        A = self.H.matMult(self.S)
        self.ksp = KspSolver()
        self.ksp.createSolver(A, self.comm)
        self.logger.info("IBM Matrices builded")
        return list(nodes)

    def collectNodes(self, cells):
        ibmNodes = self.dom.getGlobalNodesFromEntities(cells, shared=False)
        lagNodes = self.body.getTotalNodes()
        eulerCoords = self.dom.getNodesCoordinates(ibmNodes)
        ibmNodes = np.array(list(ibmNodes), dtype=np.int32)
        nodes = dict()
        maxFound = 0
        for lagNode in range(lagNodes):
            nodesFound = self.computeClose(lagNode, eulerCoords)
            coords = eulerCoords[nodesFound>0]
            nodesFound = ibmNodes[nodesFound>0]
            nodes[lagNode] = { "nodes" :nodesFound, "coords" :coords }
            if not len(nodesFound):
                raise Exception("Lag Node without Euler")
            if len(nodesFound) > maxFound:
                maxFound = len(nodesFound)
        self.eulerCoords = eulerCoords
        return nodes, maxFound

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

    # @profile
    def computeDirac(self, lagPoint, eulerCoords):
        diracs = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            d = self.body.getDiracs(dist)
            diracs.append(d)
        return diracs

    def computeClose(self, lagPoint, eulerCoords):
        close = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            dist /= self.body.getElementLong()
            if abs(dist[0]) < 2 and abs(dist[1]) < 2:
                close.append(1)
            else:
                close.append(0)
        return np.array(close)

    @staticmethod
    def computeCentroid(corners):
        return np.mean(corners, axis=0)

class ImmersedBoundaryDynamic(ImmersedBoundaryStatic):
    # @profile
    def startSolver(self):
        self.computeInitialCondition(startTime= 0.0)
        self.ts.setSolution(self.vort)
        maxSteps = self.ts.getMaxSteps()
        self.body.viewBodies()
        markNodes = self.getMarkedNodes()
        self.markZone(markNodes)
        for step in range(1,maxSteps+1):
            self.ts.step()
            time = self.ts.time
            nods = self.computeVelocityCorrection(time, NF=4)
            self.markAffectedNodes(nods)
            self.operator.Curl.mult(self.vel, self.vort)
            self.ts.setSolution(self.vort)
            if step % 10 == 0:
                self.viewer.saveData(step, time, self.vort, self.vel, self.ibmZone ,self.affectedNodes)
                self.viewer.writeXmf(self.caseName)
                self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Saved Step ")
                self.body.viewBodies()
            else:
                self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} ")

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.ibmZone = self.vort.copy()
        self.ibmZone.setName("ibm-zone")
        self.affectedNodes = self.ibmZone.copy()
        self.affectedNodes.setName("affected-nodes")
        self.body.setVelRef(self.U_ref)
        self.solveKLE(startTime, self.vort)
        self.computeVelocityCorrection(startTime, NF=1)
        self.operator.Curl.mult(self.vel, self.vort)

    def computeClose(self, lagPoint, eulerCoords):
        # TODO: mejorar el if para acotar los nodos guardados en y
        close = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            dl = self.body.getElementLong()
            dist /= dl
            if abs(dist[0]) < 2 and abs(dist[1]*dl) < (1+2*dl):
                close.append(1)
            else:
                close.append(0)
        return np.array(close)

    def markAffectedNodes(self, nodes):
        nnodes = len(nodes)
        self.affectedNodes.set(0.0)
        self.affectedNodes.setValues(nodes, [1]*nnodes, addv=False)

    def markZone(self, nodes):
        nnodes = len(nodes)
        self.ibmZone.setValues(nodes, [1]*nnodes, addv=False)

    # @profile
    def computeVelocityCorrection(self, t, NF=1):
        self.body.updateBodyParameters(t)
        affNodes = self.rebuildMatrix()
        bodyVel = self.body.getVelocity()
        # TODO Ver cuando son dos cuerpos como hacer esto!
        for i in range(NF):
            self.H.mult(self.vel, self.ibm_rhs)
            self.ksp.solve( bodyVel - self.ibm_rhs, self.virtualFlux)
            self.S.mult(self.virtualFlux, self.vel_correction)
            self.vel += self.vel_correction
        return affNodes

    def getMarkedNodes(self):
        nodes = set()
        for lagNode in self.collectedNodes.keys():
            data = self.collectedNodes[lagNode]
            # coords = data['coords']
            eulerNodes = data['nodes']
            nodes |= set(eulerNodes)
        return list(nodes)

    # @profile
    def rebuildMatrix(self):
        self.H.destroy()
        self.S.destroy()
        self.ksp.destroy()
        self.createEmptyIBMMatrix()
        nodes = self.buildIBMMatrix()
        return nodes