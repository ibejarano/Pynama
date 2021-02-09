from petsc4py.PETSc import TS, COMM_WORLD, DMPlex
from matrices.operators import Operators

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

    def rhsFunction(self, ts, time, vort, f):
        print("hello!")

    def startSolver(self):
        vort = self.__fem.dm.vortDM.getGlobalVec()
        self.__ts.solve(vort)