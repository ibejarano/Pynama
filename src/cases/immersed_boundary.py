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
from domain.immersed_body import Circle, Line
import matplotlib.pyplot as plt
import yaml
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
        else:
            self.h /= 4
        self.body = self.createBody(inputData['body'])

    def startSolver(self):
        self.computeInitialCondition(startTime= 0.0)
        self.ts.setSolution(self.vort)
        cds = list()
        clifts = list()
        times = list()
        maxSteps = self.ts.getMaxSteps()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel(r'Coef. de arrastre $C_D$', color='red')
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:blue'
        ax2.set_ylabel(r'Coef. de empuje $C_{L}$', color='blue')  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        for i in range(maxSteps):
            self.ts.step()
            step = self.ts.getStepNumber()
            time = self.ts.time
            dt = self.ts.getTimeStep()
            qx , qy, fs = self.computeVelocityCorrection(NF=4)
            cd, cl = self.computeDragForce(qx / dt, qy / dt)
            cds.append(cd)
            clifts.append(cl)
            times.append(time)
            self.operator.Curl.mult(self.vel, self.vort)
            # self.solveKLE(time, self.vort)
            self.viewer.saveVec(self.vel, timeStep=step)
            self.viewer.saveVec(self.vort, timeStep=step)
            self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort])
            self.logger.info(f"Nodos arafue {fs}  Converged: Step {step:4} | Time {time:.4e} | Cd {cd:.6f} | Cl {cl:.3f}")
            self.ts.setSolution(self.vort)
            self.viewer.writeXmf("ibm-static")
            if i % 50 == 0:
                ax1.clear()
                ax2.clear()
                ax1.set_ylim(0,3.5)
                ax2.set_ylim(0,3.5)
                ax1.plot(times, cds, color="red")
                ax2.plot(times, clifts, color="blue")
                plt.pause(0.0001)
            if time > self.ts.getMaxTime():
                break
        # plt.show()
        data = {"times": times, "cd": cds, "cl": clifts}
        with open('data.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        nombre = "re40"
        plt.savefig(f"cdsclsVtime-{nombre}.png")

    def computeDragForce(self, fd, fl):
        U = self.U_ref
        denom = 0.5 * (U**2)
        cd = fd/denom
        return float(cd), float(fl/denom)

    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)
        self.solveKLE(startTime, self.vort)
        self.computeVelocityCorrection(NF=2)
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

    def computeVelocityCorrection(self, NF=1):
        fx = 0
        fy = 0
        aux = self.vel.copy()
        bodyVel = self.body.getVelocity()
        for i in range(NF):
            self.H.mult(self.vel, self.ibm_rhs)
            self.ksp.solve(bodyVel - self.ibm_rhs, self.virtualFlux)
            self.S.mult(self.virtualFlux, self.vel_correction)
            # fx += fx_part
            # fy += fy_part
            self.vel += self.vel_correction
            aux = self.virtualFlux
            # self.H.multTranspose(self.ibm_rhs, aux)
            fx_part, fy_part, fs = self.body.computeForce(aux)
            fx += fx_part
            fy += fy_part
        return abs(fx)*self.h**2, abs(fy)*self.h**2, fs

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
            totNNZ = 0
            for i in dirac:
                if i>0:
                    totNNZ +=1
            for dof in range(self.dim):
                self.H.setValues(lagNode*self.dim+dof, eulerIndices[dof::self.dim], dirac)
            self.body.setEulerNodes(lagNode, totNNZ)

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
