import sys
import petsc4py
petsc4py.init(sys.argv)
from cases.base_problem import FreeSlip
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer
# from viewer.plotter import DualAxesPlotter
from solver.ksp_solver import KspSolver
from domain.immersed_body import Circle, Line, BodiesContainer
# import matplotlib.pyplot as plt
import yaml
# ibm imports
from math import sqrt, sin, pi, ceil, erf, exp, cos, radians
from common.timer import Timer

class ImmersedBoundaryStatic(FreeSlip):
    def setUp(self):
        super().setUp()
        self.setUpBodies()
        cells = self.getAffectedCells(10)
        self.collectedNodes, self.maxNodesPerLag = self.collectNodes(cells)
        self.totalEulerNodes = len(self.dom.getAllNodes())
        self.createEmptyIBMMatrix()
        self.mNodes = self.buildIBMMatrix()

    def readBoundaryCondition(self, bc):
        self.nu = self.mu / self.rho
        try:
            re = bc['constant']['re']
            self.logger.info(f"Material Properties: density={self.rho} ; viscosity={self.mu}")
            directionAngle = bc['constant']['direction']
            angleRadian = radians(directionAngle)
            L = eval(bc['constant']['longRef'])
            velRef = re*(self.mu/self.rho) / L
            self.U_ref = velRef
            self.cteValue = [cos(angleRadian)*velRef,sin(angleRadian)*velRef]
            self.re = re
        except:
            vel = bc['constant']['vel']
            self.U_ref = vel[0]
            self.cteValue = [self.U_ref, 0]
            self.re = self.U_ref / self.nu
        self.logger.info(f"Velocity Free Stream: {self.cteValue}")

    def setUpDomain(self):
        super().setUpDomain()
        if self.meshType == 'box-mesh':
            nelem = self.nelem
            lower, upper = self.dom.getBoundingBox()
            height = upper[1]  -  lower[0]
            self.h = (height/nelem[0]) / (self.ngl-1)
        elif self.meshType == 'gmsh-file':
            self.h = self.config['domain']['h-min'] / (self.ngl-1)
        else:
            raise Exception("Mesh not defined")

    def setUpBodies(self):
        try:
            assert 'bodies' in self.config
            bodies = self.config['bodies']
            self.body = BodiesContainer(bodies)
            self.logger.info(f"Node separation: {self.h}")
            self.body.createBodies(self.h)
            self.body.setVelRef(self.U_ref)
            # self.body.viewBodies()
        except AssertionError:
            raise Exception("Bodies not defined")

    def buildOperators(self):
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        localOperators = self.elemType.getElemKLEOperators(cornerCoords)
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            # cornerCoords = self.dom.getCellCornersCoords(cell)
            # localOperators = self.elemType.getElemKLEOperators(cornerCoords)
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            self.operator.setValues(localOperators, nodes)
        self.operator.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"Operators Matrices builded")

    def startSolver(self):
        self.computeInitialCondition()
        self.ts.setSolution(self.vort)
        saveSteps = self.config['save-n-steps']
        self.body.updateVelocity()
        cds = list()
        clifts = list()
        times = list()
        dts = list()
        steps = list()
        timer = Timer()
        elapsedTimes = list()
        maxSteps = self.ts.getMaxSteps()
        markNodes = self.getMarkedNodes()
        self.markAffectedNodes(self.mNodes)
        self.markZone(markNodes)
        for i in range(maxSteps+1):
            timer.tic()
            self.ts.step()
            step = self.ts.getStepNumber()
            time = self.ts.time
            sol = self.ts.getSolution()
            self.solveKLE(time, sol)
            self.computeVelocityCorrection()
            self.operator.Curl.mult(self.vel, self.vort)
            dt = self.ts.getTimeStep()
            self.ts.setSolution(self.vort)
            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | DT: {dt:.4e}  ")
            if time > self.ts.getMaxTime():
                break
            elif i % saveSteps == 0:
                self.viewer.saveData(step, time, self.vort, self.vel, self.ibmZone ,self.affectedNodes)
                self.viewer.writeXmf(self.caseName)
                self.H.mult(self.vel, self.ibm_rhs)
            # if i % int(saveSteps/20) == 0:
            #     cd, cl = self.computeDragForce(dt)
            #     cds.append(cd)
            #     clifts.append(cl)
            #     times.append(time)
            #     dl = self.body.getElementLong()
            #     dts.append(dt)
            #     steps.append(step)
            #     elTimes = timer.toc()
            #     elapsedTimes.append(elTimes.total_seconds())
            #     data = {
            #             "dh": self.h,
            #             "dl": dl,
            #             "lagPoints":self.body.getTotalNodes(),
            #             "eulerNodes": self.vort.getSizes()[0] ,
            #             "ngl": self.ngl,
            #             "times": times, 
            #             "cd": cds,
            #             "cl": clifts,
            #             "dt": dts,
            #             "steps": steps,
            #             "elapsedTimes": elapsedTimes
            #             }
            #     self.viewer.writeYaml(self.caseName, data)

                

    def computeDragForce(self, dt):
        U = self.U_ref
        denom = 0.5 * (U**2)

        # numOfBodies = self.body.bodyNumbers()
        # if numOfBodies == 1:
        #     forces = self.vel_correction
        #     fx_velCorr = forces[::2].sum() / dt
        #     fy_velCorr = forces[1::2].sum() / dt
        #     return float(fx_velCorr/denom), float(fy_velCorr/denom)

        # else:
        forces_x, forces_y = self.body.computeForce(self.virtualFlux, denom*dt)
        return forces_x, forces_y


    def computeInitialCondition(self):
        startTime = self.ts.getTime()
        self.vort.set(0.0)
        self.ibmZone = self.vort.copy()
        self.ibmZone.setName("ibm-zone")
        self.affectedNodes = self.ibmZone.copy()
        self.affectedNodes.setName("affected-nodes")
        self.body.setVelRef(self.U_ref)
        self.solveKLE(startTime, self.vort)
        self.computeVelocityCorrection(t=startTime)
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

    def applyBoundaryConditions(self, time):
        self.vel.set(0.0)
        velDofs = [nodes*self.dim + dof for nodes in self.bcNodes for dof in range(self.dim)]
        self.vel.setValues(velDofs, np.tile(self.cteValue, len(self.bcNodes)))
        self.vort.setValues( self.bcNodes, np.zeros(len(self.bcNodes)) , addv=False )

    def computeVelocityCorrection(self, **kwargs):
        bodyVel = self.body.getVelocity()
        self.H.mult(self.vel, self.ibm_rhs)
        self.ibm_rhs.axpy(-1, bodyVel)
        self.ksp.solve( -self.ibm_rhs, self.virtualFlux)
        self.S.mult(self.virtualFlux, self.vel_correction)
        self.vel += self.vel_correction

    def createEmptyIBMMatrix(self):
        rows = self.body.getTotalNodes() * self.dim
        cols = self.totalEulerNodes * self.dim
        d_nnz_D = self.maxNodesPerLag
        o_nnz_D = 0

        self.H = PETSc.Mat().createAIJ(size=(rows, cols), 
            nnz=(d_nnz_D, o_nnz_D), 
            comm=self.comm)
        self.H.setUp()

    def buildIBMMatrix(self):
        nodes = set()
        # self.dirs = list()
        # self.markedNodes = list()
        for lagNode in self.collectedNodes.keys():
            data = self.collectedNodes[lagNode]
            coords = data['coords']
            eulerNodes = data['nodes']
            eulerIndices = data['indices']
            dirac = np.array(self.computeDirac(lagNode, coords))
            for dof in range(self.dim):
                self.H.setValues(lagNode*self.dim+dof, eulerIndices[dof::self.dim], dirac)

            # if lagNode == 0 or lagNode == 43:
            nodes |= set(eulerNodes[dirac > 0])
                # self.markedNodes.append(eulerNodes[dirac > 0])
                # self.dirs.append(dirac[dirac>0])

        self.H.assemble()
        self.S = self.H.copy().transpose()
        dl = self.body.getElementLong()
        self.S.scale(dl*self.h)
        self.H.scale(self.h**2)
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
        indices = dict()
        maxFound = 0
        for lagNode in range(lagNodes):
            nodesFound = self.computeClose(lagNode, eulerCoords)
            coords = eulerCoords[nodesFound>0]
            nodesFound = ibmNodes[nodesFound>0]
            nodes[lagNode] = { "nodes" :nodesFound, "coords" :coords, "indices": [node*self.dim+dof for node in nodesFound for dof in range(self.dim)]}
            self.body.setEulerNodes(lagNode, len(nodesFound))
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

    def getMarkedNodes(self):
        nodes = set()
        for lagNode in self.collectedNodes.keys():
            # if lagNode == 0 or lagNode == 43 or lagNode == 86 or lagNode == 189:
            data = self.collectedNodes[lagNode]
            # coords = data['coords']
            eulerNodes = data['nodes']
            nodes |= set(eulerNodes)
        return list(nodes)

    # @profile
    def computeDirac(self, lagPoint, eulerCoords):
        diracs = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            d = self.body.getDiracs(dist, self.h)
            diracs.append(d)
        return diracs

    def computeClose(self, lagPoint, eulerCoords):
        close = list()
        coordBodyNode = self.body.getNodeCoordinates(lagPoint)
        for coords in eulerCoords:
            dist = coords - coordBodyNode
            dist /= self.body.getElementLong()
            if abs(dist[0]) < 3 and abs(dist[1]) < 3:
                close.append(1)
            else:
                close.append(0)
        return np.array(close)

    @staticmethod
    def computeCentroid(corners):
        return np.mean(corners, axis=0)
    
    def markAffectedNodes(self, nodes):
        nnodes = len(nodes)
        self.affectedNodes.set(0.0)
        self.affectedNodes.setValues(nodes, [1]*nnodes, addv=False)

    def markZone(self, nodes):
        nnodes = len(nodes)
        self.ibmZone.setValues(nodes, [1]*nnodes, addv=False)

class ImmersedBoundaryDynamic(ImmersedBoundaryStatic):

    # @profile
    def startSolver(self):
        self.computeInitialCondition()
        self.ts.setSolution(self.vort)
        maxSteps = self.ts.getMaxSteps()
        self.body.viewBodies()
        markNodes = self.getMarkedNodes()
        saveSteps = self.config['save-n-steps']
        self.markZone(markNodes)

        cds = list()
        clifts = list()
        times = list()
        dts = list()
        steps = list()
        timer = Timer()
        elapsedTimes = list()
        for step in range(1,maxSteps+1):
            timer.tic()
            self.ts.step()
            time = self.ts.time
            dt = self.ts.getTimeStep()
            nods = self.computeVelocityCorrection(time)
            self.markAffectedNodes(nods)
            self.operator.Curl.mult(self.vel, self.vort)
            self.ts.setSolution(self.vort)

            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment {dt:.3e}")
            self.body.viewBodies()


            if time > self.ts.getMaxTime():
                break
            elif step % saveSteps == 0:
                self.viewer.saveData(step, time, self.vort, self.vel, self.ibmZone ,self.affectedNodes)
                self.viewer.writeXmf(self.caseName)
                self.H.mult(self.vel, self.ibm_rhs)
            if step % int(saveSteps/10) == 0:
                cd, cl = self.computeDragForce(dt)
                cds.append(cd)
                clifts.append(cl)
                times.append(time)
                dl = self.body.getElementLong()
                dts.append(dt)
                steps.append(step)
                elTimes = timer.toc()
                elapsedTimes.append(elTimes.total_seconds())
                data = {
                        "dh": self.h,
                        "dl": dl,
                        "lagPoints":self.body.getTotalNodes(),
                        "eulerNodes": self.vort.getSizes()[0] ,
                        "ngl": self.ngl,
                        "times": times, 
                        "cd": cds,
                        "cl": clifts,
                        "dt": dts,
                        "steps": steps,
                        "elapsedTimes": elapsedTimes
                        }
                self.viewer.writeYaml(self.caseName, data)

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


    # @profile
    def computeVelocityCorrection(self, t):
        self.body.updateBodyParameters(t)
        affNodes = self.rebuildMatrix()
        bodyVel = self.body.getVelocity()
        self.H.mult(self.vel, self.ibm_rhs)
        self.ibm_rhs.axpy(-1, bodyVel)
        self.ksp.solve( - self.ibm_rhs, self.virtualFlux)
        self.S.mult(self.virtualFlux, self.vel_correction)
        self.vel += self.vel_correction

        return affNodes

    # @profile
    def rebuildMatrix(self):
        self.H.destroy()
        self.S.destroy()
        self.ksp.destroy()
        self.createEmptyIBMMatrix()
        nodes = self.buildIBMMatrix()
        return nodes