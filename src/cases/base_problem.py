import yaml
from petsc4py import PETSc
from mpi4py import MPI
import importlib
# Local packages
from domain.domain import Domain
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver
from matrices.mat_generator import Mat, Operators
from matrices.mat_ns import MatNS
from solver.ksp_solver import KspSolver
from common.timer import Timer
import logging
import numpy as np
from math import cos, sin, radians

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
        self.readMaterialData()
        self.opts = kwargs
        if "chart" in kwargs:
            self.setUpTimeSolverTest()
        elif 'time-solver' in self.config:
            self.setUpTimeSolver()

    def setUp(self):
        self.setUpDomain()
        self.setUpViewer()
        self.createMesh()
        self.bcNodes = self.dom.getBoundaryNodes()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def setUpViewer(self):
        self.viewer = Paraviewer()

    def setUpDomain(self):
        self.dom = Domain()
        self.dom.configure(self.config)
        self.dom.setOptions(**self.opts)
        self.dom.setUp()

        self.dim = self.dom.getDimension()
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6

    def readMaterialData(self):
        materialData = self.config.get("material-properties")
        self.rho = materialData['rho']
        self.mu = materialData['mu']
        self.nu = self.mu/self.rho

    def createMesh(self, saveMesh=True):
        saveDir = self.config.get("save-dir")
        self.viewer.configure(self.dim, saveDir)
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

    def createVtkFile(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('immersed-body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.dom)
        viewer.destroy()

    def evalRHS(self, ts, t, vort, f):
        """Evaluate the KLE right hand side."""
        # KLE spatial solution with vorticity given
        self.solveKLE(t, vort)
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


    def solveKLE(self, time, vort):
        self.vel.set(0.0)
        self.dom.applyBoundaryConditions(self.vel, "velocity", time, self.nu)
        self.dom.applyBoundaryConditions(self.vort, "vorticity", time, self.nu)

        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def buildKLEMats(self):
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

        assert 'initial-conditions' in self.config, "Initial conditions not defined"
        self.setUpInitialConditions()

    def setUpInitialConditions(self):
        self.logger.info("Computing initial conditions")
        initTime = self.ts.getTime()

        initialConditions = self.config['initial-conditions']
        keepCoords = initialConditions.get("keep-coords", False)

        nodes = self.dom.getAllNodes()
        inds = [ node*self.dim + dof for node in nodes for dof in range(self.dim) ]

        if 'custom-func' in initialConditions:
            customFunc = initialConditions['custom-func']
            relativePath = f".{customFunc['name']}"
            functionLib = importlib.import_module(relativePath, package='functions')

            funcVel = functionLib.velocity
            funcVort = functionLib.vorticity
            alpha = functionLib.alpha(self.nu, initTime)

            coords = self.dom.getFullCoordArray()
            arrVel = funcVel(coords, alpha)
            arrVort = funcVort(coords, alpha)

            if self.dim == 2:
                self.vort.setValues(nodes ,arrVort, addv=False)
            else:
                self.vort.setValues(inds ,arrVort, addv=False)

            self.vel.setValues(inds, arrVel, addv=False)
            if not keepCoords:
                self.dom.destroyCoordVec()

        else:
            self.dom.destroyCoordVec()
            if "velocity" in initialConditions and "vorticity" not in initialConditions:
                velArr = initialConditions['velocity']
                velArr = np.tile(velArr, len(nodes))
                self.logger.info("Computing Curl to initial velocity to get initial Vorticity")
                self.vel.setValues( inds , velArr)

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim)
        self.operator = Operators(self.dim)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()
        d_nnz_ind_op = d_nnz_ind.copy()

        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, globalIndicesDIR)
        if not self.comm.rank:
            self.logger.info(f"Empty KLE Matrices created")

        self.operator.createAll(rStart, rEnd, d_nnz_ind_op, o_nnz_ind)
        if not self.comm.rank:
            self.logger.info(f"Empty Operators created")

    def view(self):
        print(f"Case: {self.case}")
        print(f"Domain: {self.dom.view()} ")
        print(f"NGL: {self.dom.getNGL() }")

class NoSlipFreeSlip(BaseProblem):
    def setUpEmptyMats(self):
        self.mat = MatNS(self.dim)
        self.operator = Operators(self.dim)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        globalNodesDIR = self.dom.getGlobalIndicesDirichlet()
        globalNodesNS = self.dom.getGlobalIndicesNoSlip()

        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, globalNodesDIR, globalNodesNS )
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
        self.vel.set(0.0)

        self.dom.applyBoundaryConditions(self.vel, "velocity", time, self.nu)
        # self.dom.applyBoundaryConditions(self.vort, "vorticity", time, self.nu)

        self.solverFS( self.mat.Rw * vort + self.mat.Rwfs * vort\
             + self.mat.Krhsfs * self.vel , self.velFS)

        self.dom.applyBoundaryConditions(self.velFS, "velocity", time, self.nu)
        vort = self.operator.Curl * self.velFS
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def buildKLEMats(self):
        indices2one = set() 
        indices2onefs = set()
        cellStart , cellEnd = self.dom.getLocalCellRange()

        globalTangIndicesNS = self.dom.getTangDofs()
        globalNormalIndicesNS = self.dom.getNormalDofs()
        for cell in range(cellStart, cellEnd):
            nodes , inds , localMats = self.dom.computeLocalKLEMats(cell)
            locK, locRw, locRd = localMats
            indicesVel, indicesW = inds

            indicesVelSet = set(indicesVel)
            normalDofs = globalNormalIndicesNS & indicesVelSet
            tangentialDofs = globalTangIndicesNS & indicesVelSet
            tangentialDofs -= normalDofs

            gldofSetFSNS = list(normalDofs)
            gldofFreeFSSetNS = list(tangentialDofs)
            gldofFree = list(indicesVelSet - normalDofs - tangentialDofs)

            dofFree = [ indicesVel.index(i) for i in gldofFree ]
            locNormalDof = [ indicesVel.index(i) for i in normalDofs ]
            locTangDof = [ indicesVel.index(i) for i in tangentialDofs ]

            dofFreeFSSetNS = locTangDof
            dofSetFSNS = locNormalDof
            dof2beSet = list(set(dofFreeFSSetNS) | set(dofSetFSNS))
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]

            if normalDofs | tangentialDofs:
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


class BaseProblemTest(BaseProblem):

    def generateExactVecs(self, time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.velFunction(coords, self.nu, t=time)
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t=time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        return exactVel, exactVort

    def solveKLETests(self, steps=10):
        self.logger.info("Running KLE Tests")
        startTime = self.ts.getTime()
        endTime = self.ts.getMaxTime()
        times = np.linspace(startTime, endTime, steps)
        viscousTimes=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        times = [(tau**2)/(4*self.nu) for tau in viscousTimes]
        nodesToPlot, coords = self.dom.getNodesOverline("x", 0.5)
        plotter = Plotter(r"$\frac{u}{U}$" , r"$\frac{y}{Y}$")
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            self.operator.Curl.mult( exactVel , self.vort )
            self.viewer.saveData(step, time, self.vel, self.vort, exactVel, exactVort)
            exact_x , _ = self.dom.getVecArrayFromNodes(exactVel, nodesToPlot)
            calc_x, _ = self.dom.getVecArrayFromNodes(self.vel, nodesToPlot)
            # plotter.updatePlot(exact_x, [{"name": fr"$\tau$ = ${viscousTimes[step]}$" ,"data":coords}], step )
            # plotter.scatter(calc_x , coords, "calc")
            # plotter.plt.pause(0.001)
            self.logger.info(f"Saving time: {time:.1f} | Step: {step}")
        # plotter.plt.legend()
        # plotter.show()
        self.viewer.writeXmf(self.caseName)

    def generateExactOperVecs(self,time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactConv = exactVort.copy()
        exactDiff = exactVort.copy()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        exactConv.setName(f"{self.caseName}-exact-convective")
        exactDiff.setName(f"{self.caseName}-exact-diffusive")
        allNodes = self.dom.getAllNodes()
        # generate a new function with t=constant and coords variable
        fvel_coords = lambda coords: self.velFunction(coords, self.nu, t = time)
        fvort_coords = lambda coords: self.vortFunction(coords, self.nu, t = time)
        fconv_coords = lambda coords: self.convectiveFunction(coords, self.nu, t = time)
        fdiff_coords = lambda coords: self.diffusiveFunction(coords, self.nu, t = time)
        exactVel = self.dom.applyFunctionVecToVec(allNodes, fvel_coords, exactVel, self.dim)
        exactVort = self.dom.applyFunctionVecToVec(allNodes, fvort_coords, exactVort, self.dim_w)
        exactConv = self.dom.applyFunctionVecToVec(allNodes, fconv_coords, exactConv, self.dim_w )
        exactDiff = self.dom.applyFunctionVecToVec(allNodes, fdiff_coords, exactDiff, self.dim_w )
        return exactVel, exactVort, exactConv, exactDiff

    def OperatorsTests(self, viscousTime=1):
        time = (viscousTime**2)/(4*self.nu)
        self.applyBoundaryConditions(time, boundaryNodes)
        step = 0
        exactVel, exactVort, exactConv, exactDiff = self.generateExactOperVecs(time)
        #exactDiff.scale(2*self.mu)
        self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
        convective = self.getConvective(exactVel, exactConv)
        convective.setName("convective")
        diffusive = self.getDiffusive(exactVel, exactDiff)
        diffusive.setName("diffusive")
        self.operator.Curl.mult(exactVel, self.vort)
        self.viewer.saveData(step, time, self.vel, self.vort, exactVel, exactVort,exactConv,exactDiff,convective,diffusive )
        self.viewer.writeXmf(self.caseName)
        self.operator.weigCurl.reciprocal()
        err = convective - exactConv
        errorConv = sqrt((err * err ).dot(self.operator.weigCurl))
        err = diffusive - exactDiff
        errorDiff = sqrt((err * err ).dot(self.operator.weigCurl))
        err = self.vort - exactVort
        errorCurl = sqrt((err * err ).dot(self.operator.weigCurl))
        self.logger.info("Operatores Tests")
        return errorConv, errorDiff, errorCurl

    def getConvective(self, exactVel, exactConv):
        convective = exactConv.copy()
        self.computeVtensV()
        aux=self.vel.copy()
        self.operator.DivSrT.mult(self._VtensV, aux)
        self.operator.Curl.mult(aux,convective)
        return convective

    def getDiffusive(self, exactVel, exactDiff):
        diffusive = exactDiff.copy()
        self.operator.SrT.mult(exactVel, self._Aux1)
        aux=self.vel.copy()
        self._Aux1 *= (2.0 * self.mu)
        self.operator.DivSrT.mult(self._Aux1, aux)
        aux.scale(1/self.rho)
        self.operator.Curl.mult(aux,diffusive)
        return diffusive

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