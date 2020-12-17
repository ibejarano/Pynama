import yaml
from petsc4py import PETSc
from mpi4py import MPI
# Local packages
from domain.dmplex import DMPlexDom, Domain
from domain.elements.spectral import Spectral
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver
from matrices.mat_generator import Mat, Operators
from matrices.mat_ns import MatNS
from solver.ksp_solver import KspSolver
from common.timer import Timer
from math import sqrt
import logging
import numpy as np

class BaseProblem(object):
    def __init__(self, config,**kwargs):
        """
        comm: MPI Communicator
        """
        self.comm = MPI.COMM_WORLD
        self.timerTotal= Timer()
        self.timerTotal.tic()
        self.timer = Timer()
        if 'case' in kwargs:
            case = kwargs['case']
        else:
            case = PETSc.Options().getString('case', 'uniform')
        self.config = config
        self.logger = logging.getLogger(f"[{self.comm.rank}] {self.config.get('name')}")
        self.case = case
        self.caseName = self.config.get("name")
        self.readDomainData(kwargs)
        self.readMaterialData()
        self.opts = kwargs
        if "chart" in kwargs:
            self.setUpTimeSolverTest()
        elif 'time-solver' in self.config:
            self.setUpTimeSolver()

        if 'boundary-conditions' in self.config:
            boundaryConditions = self.config.get("boundary-conditions")
            self.readBoundaryCondition(boundaryConditions)

    def setUp(self):
        self.setUpDomain()
        self.createMesh()
        self.bcNodes = self.dom.getBoundaryNodes()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def setUpDomain(self):
        domain = self.config.get("domain")
        self.dom = None
        self.dom = Domain(domain, **self.opts)

        self.dim = self.dom.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

        self.dom.setUp()

    def readMaterialData(self):
        materialData = self.config.get("material-properties")
        self.rho = materialData['rho']
        self.mu = materialData['mu']
        self.nu = self.mu/self.rho

    def readDomainData(self, kwargs):
        domain = self.config.get("domain")
        if "nelem" in kwargs:
            self.nelem = kwargs['nelem']
        elif "box-mesh" in domain:
            self.nelem = domain['box-mesh']['nelem']
            self.lower = domain['box-mesh']['lower']
            self.upper = domain['box-mesh']['upper']

        else:
            raise Exception("No Gmsh Implemented")

        self.dim = len(self.nelem)
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        if "ngl" in kwargs:
            self.ngl = kwargs['ngl']
        else:
            self.ngl = domain['ngl']

    def createMesh(self, saveMesh=True):
        saveDir = self.config.get("save-dir")
        self.viewer = Paraviewer(self.dim ,self.comm, saveDir)
        self.dom.computeFullCoordinates()
        if saveMesh:
            self.viewer.saveMesh(self.dom.getFullCoordVec())
        if not self.comm.rank:
            self.logger.info(f"Mesh created")

    # @profile
    def buildOperators(self):
        cellStart, cellEnd = self.dom.getLocalCellRange()
        for cell in range(cellStart, cellEnd):
            nodes, localOperators = self.dom.computeLocalOperators(cell)
            self.operator.setValues(localOperators, nodes)
        self.operator.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"Operators Matrices builded")

    def setUpTimeSolver(self):
        options = self.config.get("time-solver")
        self.ts = TsSolver(self.comm)
        sTime = options['start-time']
        eTime = options['end-time']
        maxSteps = options['max-steps']
        self.ts.setUpTimes(sTime, eTime, maxSteps)
        self.ts.initSolver(self.evalRHS, self.convergedStepFunction)

    def setUpTimeSolverTest(self):
        options = self.config.get("time-solver")
        self.ts = TsSolver(self.comm)
        sTime = options['start-time']
        eTime = options['end-time']
        maxSteps = options['max-steps']
        self.ts.setUpTimes(sTime, eTime, maxSteps)
        self.saveError2 = []
        self.saveError8 = []
        self.saveStep = []
        self.saveTime = []
        self.ts.initSolver(self.evalRHS, self.convergedStepFunctionKLET)
        
    def createNumProcVec(self, step):
        proc = self.vort.copy()
        proc.setName("num proc")
        beg, end = proc.getOwnershipRange() 
        for i in range(beg,end):
            proc.setValue(i, self.comm.rank)
        proc.assemble()
        self.createVtkFile()
        return proc

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        incr = ts.getTimeStep()
        self.viewer.saveData(step, time, self.vel, self.vort)
        # self.viewer.newSaveVec([self.vel, self.vort], step)
        self.viewer.writeXmf(self.caseName)
        if not self.comm.rank:
            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")

    def convergedStepFunctionKLET(self, ts):
        time = ts.time
        step = ts.step_number
        self.solveKLE(time, self.vort)
        incr = ts.getTimeStep()
        # self.viewer.newSaveVec([self.vel, self.vort], step)
        exactVel, exactVort = self.generateExactVecs(time)
        errorVel = exactVel - self.vel
        errorVort = exactVort - self.vort
        self.saveError2.append((errorVel).norm(norm_type=2))
        self.saveError8.append((errorVel).norm(norm_type=3))
        self.saveTime.append(time)
        self.saveStep.append(step)
        errorVort.setName("ErrorVort")
        errorVel.setName("ErrorVel")
        exactVort.setName("ExactVort")
        exactVel.setName("ExactVel")
        self.viewer.saveData(step, time, self.vel, self.vort, exactVel, exactVort,errorVel,errorVort)
        self.viewer.writeXmf(self.caseName)        
        if not self.comm.rank:
            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")

    def createVtkFile(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('immersed-body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.dom)
        viewer.destroy()

    def evalRHS(self, ts, t, Vort, f):
        """Evaluate the KLE right hand side."""
        # KLE spatial solution with vorticity given
        self.solveKLE(t, Vort)
        # FIXME: Generalize for dim = 3 also
        self.computeVtensV()
        # self._Aux1 = self.SrT * self._Vel
        self.operator.SrT.mult(self.vel, self._Aux1)

        # _Aux1 = 2*Mu * S - rho * Vvec ^ VVec
        self._Aux1 *= (2.0 * self.mu)
        self._Aux1.axpy(-1.0 * self.rho, self._VtensV)

        # FIXME: rhs should be created previously or not?
        rhs = self.vel.duplicate()
        # RHS = Curl * Div(SrT) * 2*Mu * S - rho * Vvec ^ VVec
            # rhs = (self.DivSrT * self._Aux1) / self.rho
        self.operator.DivSrT.mult(self._Aux1, rhs)
        rhs.scale(1/self.rho)

        self.operator.Curl.mult(rhs, f)

    def computeVtensV(self, vec=None):
        if vec is not None:
            arr = vec.getArray()
        else:
            arr = self.vel.getArray()

        sK, eK = self.mat.K.getOwnershipRange()
        ind = np.arange(int(sK*self.dim_s/self.dim), int(eK*self.dim_s/self.dim), dtype=np.int32)
        v_x = arr[::self.dim]
        v_y = arr[1::self.dim]
        self._VtensV.setValues(ind[::self.dim_s], v_x**2 , False)
        self._VtensV.setValues(ind[1::self.dim_s], v_x * v_y , False)
        self._VtensV.setValues(ind[2::self.dim_s], v_y**2 , False)
        if self.dim == 3:
            v_z = arr[2::self.dim]
            self._VtensV.setValues(ind[3::self.dim_s], v_y * v_z , False)
            self._VtensV.setValues(ind[4::self.dim_s], v_z**2 , False)
            self._VtensV.setValues(ind[5::self.dim_s], v_z * v_x , False)
        self._VtensV.assemble()

    def startSolver(self):
        initTime = self.ts.getTime()
        self.computeInitialCondition(startTime=initTime)
        self.ts.solve(self.vort)

    def solveKLE(self, time, vort):
        pass

    def buildKLEMats(self):
        pass

    def computeInitialCondition(self, startTime):
        pass

    def applyBoundaryConditions(self, time):
        pass

    def readBoundaryCondition(self):
        pass

    def setUpSolver(self):
        self.solver = KspSolver()
        self.solver.createSolver(self.mat.K, self.comm)
        self.vel = self.mat.K.createVecRight()
        self.vel.setName("velocity")
        self.vort = self.mat.Rw.createVecRight()
        self.vort.setName("vorticity")
        self.vort.set(0.0)

        sK, eK = self.mat.K.getOwnershipRange()
        locRowsK = eK - sK

        self._VtensV = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)
        self._Aux1 = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)

    def setUpEmptyMats(self):
        self.mat = None
        self.operator = None

    def view(self):
        print(f"Case: {self.case}")
        print(f"Domain: {self.dom.view()} ")
        print(f"NGL: {self.dom.getNGL() }")

class NoSlipFreeSlip(BaseProblem):
    def setUpEmptyMats(self):
        self.mat = MatNS(self.dim, self.comm)
        self.operator = Operators(self.dim, self.comm)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        self.globalNodesDIR = self.dom.getGlobalIndicesDirichlet()
        globalNodesNS = self.dom.getGlobalIndicesNoSlip()
        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, self.globalNodesDIR, globalNodesNS )
        if not self.comm.rank:
            self.logger.info(f"Empty KLE Matrices created")

        self.operator.createAll(rStart, rEnd, d_nnz_ind, o_nnz_ind)
        if not self.comm.rank:
            self.logger.info(f"Empty Operators created")

    def setUpSolver(self):
        super().setUpSolver()
        self.solverFS = KspSolver()
        self.solverFS.createSolver(self.mat.K + self.mat.Kfs, self.comm) 
        self.velFS = self.vel.copy()

    def solveKLE(self, time, vort):
        self.applyBoundaryConditions()
        self.solverFS( self.mat.Rw * vort + self.mat.Rwfs * vort\
             + self.mat.Krhsfs * self.vel , self.velFS)
        self.applyBoundaryConditionsFS()
        vort = self.operator.Curl * self.velFS
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def buildKLEMats(self):
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        boundaryNodesNS = self.mat.globalIndicesNS
        boundaryNodesDIR =  self.mat.globalIndicesDIR
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)
        indices2one = set()
        indices2onefs = set()
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
            nodeBCintersectNS = boundaryNodesNS & set(nodes)
            nodeBCintersectDIR = boundaryNodesDIR & set(nodes)
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions
            # for node in nodeBCintersect:
            #     localBoundaryNode = nodes.index(node)
            #     nsNorm = set()
            #     coord=self.dom.getNodesCoordinates([node])[0]
            #     for i in range(self.dim):
            #         if (coord[i]==self.upper[i]) or (coord[i]==self.lower[i]):
            #             nsNorm.add(i)
            #     dofSetFSNS.update([localBoundaryNode*self.dim + d
            #                            for d in nsNorm])
            #     dofFreeFSSetNS.update([localBoundaryNode*self.dim + d
            #                             for d in (set(range(self.dim)) - nsNorm)])

            if nodeBCintersectNS:
                borderNodes, normals = self.dom.getBorderNodesWithNormal(cell, nodeBCintersectNS)
                for i, globBorderNodes in enumerate(borderNodes):
                    tangentialDofs = list(range(self.dim))
                    tangentialDofs.pop(normals[i])
                    localNodes = [ nodes.index(node) for node in globBorderNodes ]
                    normalDof = normals[i]
                    # print(f"{[self.comm.rank]} {nodes = } {globBorderNodes = } {normals = } {localNodes = } ")
                    dofSetFSNS.update( [locNode * self.dim + normalDof for locNode in localNodes ]  )
                    dofFreeFSSetNS.update( [locNode * self.dim + dof for locNode in localNodes for dof in tangentialDofs] )
                dofFreeFSSetNS -= dofSetFSNS

            if nodeBCintersectDIR:
                borderNodes, _ = self.dom.getBorderNodesWithNormal(cell, nodeBCintersectDIR)
                for i, globBorderNodes in enumerate(borderNodes):
                    localNodes = [ nodes.index(node) for node in globBorderNodes ]
                    dofSetFSNS.update( [locNode*self.dim + dof for locNode in localNodes for dof in range(self.dim)])

            dofFree = list(set(range(len(indicesVel)))
                           - dofFreeFSSetNS - dofSetFSNS)
            dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
            dofFreeFSSetNS = list(dofFreeFSSetNS)
            dofSetFSNS = list(dofSetFSNS)
            gldofFreeFSSetNS = [indicesVel[ii] for ii in dofFreeFSSetNS]
            gldofSetFSNS = [indicesVel[ii] for ii in dofSetFSNS]
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            
            if nodeBCintersectNS | nodeBCintersectDIR:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)
                self.mat.Kfs.setValues(gldofFreeFSSetNS, gldofFree,
                                    locK[np.ix_(dofFreeFSSetNS, dofFree)],
                                    addv=True)

                self.mat.Kfs.setValues(gldofFree, gldofFreeFSSetNS,
                                    locK[np.ix_(dofFree, dofFreeFSSetNS)],
                                    addv=True)

                self.mat.Kfs.setValues(
                    gldofFreeFSSetNS, gldofFreeFSSetNS,
                    locK[np.ix_(dofFreeFSSetNS, dofFreeFSSetNS)],
                    addv=True)

                # Indices where diagonal entries should be reduced by 1
                indices2onefs.update(gldofFreeFSSetNS)

                self.mat.Rwfs.setValues(gldofFreeFSSetNS, indicesW,
                                    locRw[dofFreeFSSetNS, :], addv=True)

                self.mat.Rdfs.setValues(gldofFreeFSSetNS, nodes,
                                    locRd[dofFreeFSSetNS, :], addv=True)
                self.mat.Krhsfs.setValues(
                        gldofFreeFSSetNS, gldofSetFSNS,
                        - locK[np.ix_(dofFreeFSSetNS, dofSetFSNS)], addv=True)
                self.mat.Krhsfs.setValues(
                        gldofFree, gldofSetFSNS,
                        - locK[np.ix_(dofFree, dofSetFSNS)], addv=True)
                for indd in gldofSetFSNS:
                        self.mat.Krhsfs.setValues(indd, indd, 0, addv=True)

            self.mat.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.mat.K.setValues(indd, indd, 0, addv=True)

            self.mat.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.mat.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)
        self.mat.assembleAll()
        self.mat.setIndices2One(indices2one)

        for indd in indices2onefs:
            self.mat.Kfs.setValues(indd, indd, -1, addv=True)

        self.mat.Kfs.assemble()
        self.mat.Rwfs.assemble()
        self.mat.Rdfs.assemble()
        self.mat.Krhsfs.assemble()

        for indd in (indices2one - indices2onefs):
            self.mat.Krhsfs.setValues(indd, indd, 1, addv=False)
        self.mat.Krhsfs.assemble()

        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")

class FreeSlip(BaseProblem):
    def generateExactVecs(self, time):
        return 0, 0

    def buildMatrices(self):
        pass

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        self.operator = Operators(self.dim, self.comm)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()

        d_nnz_ind_op = d_nnz_ind.copy()

        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, globalIndicesDIR)
        if not self.comm.rank:
            self.logger.info(f"Empty KLE Matrices created")

        self.operator.createAll(rStart, rEnd, d_nnz_ind_op, o_nnz_ind)
        if not self.comm.rank:
            self.logger.info(f"Empty Operators created")

    def solveKLE(self, time, vort):
        self.applyBoundaryConditions(time)
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def getKLEError(self, viscousTimes=None ,startTime=0.0, endTime=1.0, steps=10):
        try:
            assert viscousTimes !=None
        except:
            viscousTimes = np.arange(startTime, endTime, (endTime - startTime)/steps)

        times = [(tau**2)/(4*self.nu) for tau in viscousTimes]
        errors = list()
        for time in times:
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            error = (exactVel - self.vel).norm(norm_type=2)
            errors.append(error)
        return errors

    def buildKLEMats(self):
        # indices2one = set() 
        # cornerCoords = self.dom.getCellCornersCoords(cell=0)
        # locK, locRw, _ = self.elemType.getElemKLEMatrices(cornerCoords)
        globalBCNodes = self.bcNodes

        cellStart , cellEnd = self.dom.getLocalCellRange()

        for cell in range(cellStart, cellEnd):
            nodes , inds , localMats = self.dom.computeLocalKLEMats(cell)
            locK, locRw, _ = localMats
            indicesVel, indicesW = inds
            
            nodeBCintersect = set(globalBCNodes) & set(nodes)
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions

            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                # FIXME : No importa el bc, #TODO cuando agregemos NS si importa
                for dof in range(self.dim):
                    dofSetFSNS.add(localBoundaryNode*self.dim + dof)

            dofFree = list(set(range(len(indicesVel)))
                           - dofFreeFSSetNS - dofSetFSNS)
            dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
            dofFreeFSSetNS = list(dofFreeFSSetNS)
            dofSetFSNS = list(dofSetFSNS)
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            
            if nodeBCintersect:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                # indices2one.update(gldof2beSet)
                # print(indices2one)
                # FIXME: is the code below really necessary?
                # for indd in gldof2beSet:
                #     self.mat.Krhs.setValues(indd, indd, 0, addv=True)

            self.mat.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.mat.K.setValues(indd, indd, 0, addv=True)

            self.mat.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

        self.mat.setIndices2One(self.mat.globalIndicesDIR)
        self.mat.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")