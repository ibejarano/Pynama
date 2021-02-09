from petsc4py.PETSc import TS, COMM_WORLD, DMPlex
from matrices.operators import Operators
import numpy as np

class TSSolver(TS):
    rk_types = ["3", "5f", "5bs"]
    def __init__(self, comm=COMM_WORLD):
        self.create(comm)
        self.setProblemType(self.ProblemType.NONLINEAR)
        self.setEquationType(self.EquationType.ODE_EXPLICIT)
        self.setType('rk')
        self.setRKType('5bs')

    def setUpTimes(self, sTime, eTime, steps):
        self.setTime(sTime)
        self.setMaxTime(eTime)
        self.setMaxSteps(steps)
        self.setExactFinalTime(self.ExactFinalTime.MATCHSTEP)
        # self.setExactFinalTime(self.ExactFinalTime.INTERPOLATE)
        # Sundials doesn't support MATCHSTEP (-ts_exact_final_time INTERPOLATE)

    def initSolver(self, rhsFunction, operators, fem):
        opers = { 'curl': operators.Curl, 'div': operators.DivSrT, 'srt': operators.SrT, 'fem': fem}
        self.setRHSFunction(rhsFunction, None , kargs=opers)
        self.setPostStep(fem.saveStep)
        self.setFromOptions()

class TimeStepping:
    rk_types = ["3", "5f", "5bs"]
    def __init__(self, comm=COMM_WORLD):
        self.__ts = TS().create(comm)
        self.__ts.setProblemType(TS.ProblemType.NONLINEAR)
        self.__ts.setEquationType(TS.EquationType.ODE_EXPLICIT)
        self.__ts.setType('rk')
        self.__ts.setRKType('5bs')

    def setFem(self, fem):
        self.__fem = fem
        dmVort = self.__fem.dm.vortDM
        self.__ts.setDM(dmVort)

    def setUp(self):
        self.setUpTimes()
        self.createOperators()
        self.initSolver()

    def setUpTimes(self):
        config = self.__fem.config.get('time-solver')
        startTime = config.get('start-time', None)
        endTime = config.get('end-time', None)
        maxSteps = config.get('max-steps', None)
        self.__ts.setTime(startTime)
        self.__ts.setMaxTime(endTime)
        self.__ts.setMaxSteps(maxSteps)
        self.__ts.setExactFinalTime(TS.ExactFinalTime.MATCHSTEP)
        # self.setExactFinalTime(self.ExactFinalTime.INTERPOLATE)
        # Sundials doesn't support MATCHSTEP (-ts_exact_final_time INTERPOLATE)

    def initSolver(self):
        # opers = { 'curl': operators.Curl, 'div': operators.DivSrT, 'srt': operators.SrT, 'fem': fem}
        self.__ts.setRHSFunction(self.rhsFunction)
        self.__ts.setPostStep(self.__fem.saveStep)
        self.__ts.setFromOptions()

    def createOperators(self):
        self.operators = Operators()

        dm = self.__fem.dm.velDM
        ngl = self.__fem.dm.getNGL()

        self.operators.preallocate(config=self.__fem.config['domain']['box-mesh'] ,ngl=ngl)
        self.operators.setDM(dm)
        self.operators.setElem(self.__fem.elem)
        self.operators.assemble()

    def startSolver(self):
        startTime = self.__ts.time
        vort = self.__fem.dm.getGlobalVorticity()
        self.__fem.computeInitialConditions(vort, startTime)
        self.__ts.solve(vort)

    def rhsFunction(self, ts, time, vort, f):
        dm = ts.getDM()
        dim = dm.getDimension()
        dim_s = 3

        # 1 Setear BC a la vorticidad.
        self.__fem.computeBoundaryConditionsVort(time)
        # 2 Setear los valores internos a la vorticidad
        dm.globalToLocal(vort, self.__fem.vort)
        # 3 resolver kle y obtener velocidad
        self.__fem.solveKLE(self.__fem.vort, time)
        # 4 aplicar VtensV
        VtensV = self.operators.SrT.createVecLeft()
        startInd, endInd = self.operators.SrT.getOwnershipRange()
        ind = np.arange(startInd, endInd, dtype=np.int32)
        arr = self.__fem.vel.getArray()
        v_x = arr[::dim]
        v_y = arr[1::dim]
        VtensV.setValues(ind[::dim_s], v_x**2 , False)
        VtensV.setValues(ind[1::dim_s], v_x * v_y , False)
        VtensV.setValues(ind[2::dim_s], v_y**2 , False)
        VtensV.assemble()
        # 5 Aplicar en su orden los operadores
        aux = VtensV.duplicate() 
        self.operators.SrT.mult(self.__fem.vel, aux)

        # FIXME: Hard code mu and rho
        mu = 0.01
        rho = 0.5
        aux *= (2.0 * mu)
        aux.axpy(-1.0 * rho, VtensV)
        rhs = self.__fem.vel.duplicate()
        self.operators.DivSrT.mult(aux, rhs)
        rhs.scale(1/rho)
        locF = dm.getLocalVec()
        self.operators.Curl.mult(rhs, locF)
        dm.localToGlobal(locF, f)
        dm.restoreLocalVec(locF)