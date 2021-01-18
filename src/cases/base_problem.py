import yaml
from petsc4py import PETSc
from mpi4py import MPI
import importlib
# Local packages
from domain.domain import Domain
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver
from matrices.mat_fs import MatFS, Operators
from matrices.mat_ns import MatNS
from solver.ksp_solver import KleSolver
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
        vort = ts.getSolution()
        vel = self.solverKLE.getSolution()
        self.viewer.saveData(step, time, vel, vort)
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

        self.dom.applyBoundaryConditions(self.vort, "vorticity", t, self.nu)
        vel = self.solverKLE.getSolution()
        self.dom.applyBoundaryConditions(vel, "velocity", t, self.nu)

        if self.solverKLE.isNS():
            self.logger.info("Solving ns")
            self.solverKLE.solveFS(vort)
            velFS = self.solverKLE.getFreeSlipSolution()
            self.dom.applyBoundaryConditions(velFS, "velocity", t, self.nu)
            self.operator.Curl.mult(velFS, vort)

        self.solverKLE.solve(vort)

        self.computeVtensV(vel)
        self.operator.SrT.mult(vel, self._Aux1)
        self._Aux1 *= (2.0 * self.mu)
        self._Aux1.axpy(-1.0 * self.rho, self._VtensV)
        # FIXME: rhs should be created previously or not?
        rhs = vel.duplicate()
        self.operator.DivSrT.mult(self._Aux1, rhs)
        rhs.scale(1/self.rho)

        self.operator.Curl.mult(rhs, f)

    def computeVtensV(self, vec):
        arr = vec.getArray()
        startInd, endInd = self.operator.SrT.getOwnershipRange()
        ind = np.arange(startInd, endInd, dtype=np.int32)
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

    def setUpSolver(self):
        bcType = self.dom.getBoundaryType()
        if bcType == "FS":
            mat = MatFS()
        elif bcType =="NS":
            mat = MatNS()
        else:
            raise Exception("FSNS Mat not implemented")

        mat.setDomain(self.dom)
        mat.build()
        self.mat = mat

        self.solverKLE = KleSolver()
        self.solverKLE.setMat(mat)
        self.solverKLE.setUp()

        self.operator = mat.getOperators()

        self._VtensV = self.operator.SrT.createVecLeft()
        self._Aux1 = self._VtensV.duplicate() 

        assert 'initial-conditions' in self.config, "Initial conditions not defined"
        self.setUpInitialConditions()

    def setUpInitialConditions(self):
        self.logger.info("Computing initial conditions")
        initTime = self.ts.getTime()
        vort = self.operator.Curl.createVecLeft()
        vort.setName("vorticity")
        vel = self.solverKLE.getSolution()

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
                vort.setValues(nodes ,arrVort, addv=False)
            else:
                vort.setValues(inds ,arrVort, addv=False)

            vel.setValues(inds, arrVel, addv=False)
            if not keepCoords:
                self.dom.destroyCoordVec()

        else:
            self.dom.destroyCoordVec()
            if "velocity" in initialConditions and "vorticity" not in initialConditions:
                velArr = initialConditions['velocity']
                velArr = np.tile(velArr, len(nodes))
                self.logger.info("Computing Curl to initial velocity to get initial Vorticity")
                vel.setValues( inds , velArr)

        vort.assemble()
        vel.assemble()
        self.vort = vort
        self.ts.setSolution(vort)

        self.viewer.saveData(0, initTime, vel, vort)
        self.viewer.writeXmf(self.caseName)

    def view(self):
        print(f"Case: {self.case}")
        print(f"Domain: {self.dom.view()} ")
        print(f"NGL: {self.dom.getNGL() }")

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