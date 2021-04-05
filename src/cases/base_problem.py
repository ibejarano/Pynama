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
from solver.kle_solver import KleSolver
from common.timer import Timer
import logging
import numpy as np
from math import cos, sin, radians, sqrt

class BaseProblem(object):
    def __init__(self, config,**kwargs):
        """
        comm: MPI Communicator
        """
        self.comm = PETSc.COMM_WORLD
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
        velFS = self.solverKLE.getFreeSlipSolution()
        self.viewer.saveData(step, time, vel, vort, velFS)
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
            self.solverKLE.solveFS(self.vort)
            velFS = self.solverKLE.getFreeSlipSolution()
            self.dom.applyBoundaryConditionsNS(velFS, "velocity", t, self.nu)
            self.operator.Curl.mult(velFS, self.vort)

        self.solverKLE.solve(self.vort)

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

        else:
            if "velocity" in initialConditions and "vorticity" not in initialConditions:
                velArr = initialConditions['velocity']
                velArr = np.tile(velArr, len(nodes))
                self.logger.info("Computing Curl to initial velocity to get initial Vorticity")
                vel.setValues( inds , velArr)

        vort.assemble()
        vel.assemble()
        self.vort = vort
        # self.ts.setSolution(vort)

        # self.viewer.saveData(0, initTime, vel, vort)
        # self.viewer.writeXmf(self.caseName)

    def view(self):
        print(f"Case: {self.case}")
        print(f"Domain: {self.dom.view()} ")
        print(f"NGL: {self.dom.getNGL() }")

class BaseProblemTest(BaseProblem):

    def generateExactVecs(self,vel=None, vort=None ,time=None):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactVel.setName(f"{self.caseName}-exact-vel")
        exactVort.setName(f"{self.caseName}-exact-vort")
        allNodes = self.dom.getAllNodes()
        inds_w = [node*self.dim_w + dof for node in allNodes for dof in range(self.dim_w)]
        inds = [node*self.dim + dof for node in allNodes for dof in range(self.dim)]
        if vel and vort:
            arrVel = np.tile(vel, len(allNodes))
            arrVort =  np.tile(vort, len(allNodes))
        else:
            customFunc = self.config['tests']['custom-func']
            relativePath = f".{customFunc['name']}"
            functionLib = importlib.import_module(relativePath, package='functions')

            funcVel = functionLib.velocity
            funcVort = functionLib.vorticity
            alpha = functionLib.alpha(self.nu, time)

            coords = self.dom.getFullCoordArray()
            arrVel = funcVel(coords, alpha)
            arrVort = funcVort(coords, alpha)

        exactVort.setValues(inds_w, arrVort)
        exactVel.setValues(inds, arrVel)
        exactVel.assemble()
        exactVort.assemble()

        return exactVel, exactVort

    def solveKLETests(self, steps=10):
        self.logger.info("Running KLE Tests")
        dm = self.dom.getDM()
        startTime = self.ts.getTime()
        endTime = self.ts.getMaxTime()
        times = np.linspace(startTime, endTime, steps)
        viscousTimes=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        times = [(tau**2)/(4*self.nu) for tau in viscousTimes]
        # nodesToPlot, coords = self.dom.getNodesOverline("x", 0.5)
        for step,time in enumerate(times):
            exactVel, exactVort = self.generateExactVecs(time=time)
            self.dom.applyBoundaryConditions(self.vort, "vorticity", time, self.nu)
            vel = self.solverKLE.getSolution()
            self.dom.applyBoundaryConditions(vel, "velocity", time, self.nu)
            self.solverKLE.solve(exactVort)
            self.operator.Curl.mult( exactVel , self.vort )
            self.viewer.saveData(step, time, vel, self.vort, exactVel, exactVort)
            # exact_x , _ = dm.getVecArrayFromNodes(exactVel, nodesToPlot)
            # calc_x, _ = dm.getVecArrayFromNodes(vel, nodesToPlot)
            self.logger.info(f"Saving time: {time:.1f} | Step: {step}")

        self.viewer.writeXmf(self.caseName)

    def generateExactOperVecs(self,time):
        exactVel = self.mat.K.createVecRight()
        exactVort = self.mat.Rw.createVecRight()
        exactConv = exactVort.copy()
        exactDiff = exactVort.copy()
        exactVecs = [ exactVel, exactVort, exactConv, exactDiff]
        allNodes = self.dom.getAllNodes()
        customFuncs = self.config['tests']['custom-func']
        relativePath = f".{customFuncs['name']}"
        vecNames = 'velocity vorticity convective diffusive'.split()
        functionLib = importlib.import_module(relativePath, package='functions')
        alpha = functionLib.alpha(self.nu, time)
        coords = self.dom.getFullCoordArray()
        allNodes = self.dom.getAllNodes()

        for i, name in enumerate(vecNames):
            vec = exactVecs[i]
            vec.setName(name)
            func = functionLib.__getattribute__(name)
            values = func(coords, alpha)
            if name == 'velocity':
                inds = [node*dof + dof for node in allNodes for dof in range(self.dim)]
                vec.setValues(inds, values)
            else:
                vec.setValues(allNodes, values)
            vec.assemble()

        return exactVel, exactVort, exactConv, exactDiff

    def OperatorsTests(self, viscousTime=1):
        time = (viscousTime**2)/(4*self.nu)
        vel = self.solverKLE.getSolution()
        step = 0
        exactVel, exactVort, exactConv, exactDiff = self.generateExactOperVecs(time)
        self.dom.applyBoundaryConditions(vel, "velocity", time, self.nu)
        self.solverKLE.solve(exactVort)
        convective = self.getConvective(exactVel, exactConv)
        convective.setName("convective")
        diffusive = self.getDiffusive(exactVel, exactDiff)
        diffusive.setName("diffusive")
        self.operator.Curl.mult(exactVel, self.vort)
        self.viewer.saveData(step, time, vel, self.vort, exactVel, exactVort,exactConv,exactDiff,convective,diffusive )
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
        vel = self.solverKLE.getSolution()
        self.computeVtensV(vel)
        aux= vel.copy()
        self.operator.DivSrT.mult(self._VtensV, aux)
        self.operator.Curl.mult(aux,convective)
        return convective

    def getDiffusive(self, exactVel, exactDiff):
        diffusive = exactDiff.copy()
        vel = self.solverKLE.getSolution()
        self.operator.SrT.mult(exactVel, self._Aux1)
        aux = vel.copy()
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
            exactVel, exactVort = self.generateExactVecs(time=time)
            self.dom.applyBoundaryConditions(self.vort, "vorticity", time, self.nu)
            vel = self.solverKLE.getSolution()
            self.dom.applyBoundaryConditions(vel, "velocity", time, self.nu)
            self.solverKLE.solve(exactVort)
            error = (exactVel - vel).norm(norm_type=2)
            errors.append(error)
        return errors