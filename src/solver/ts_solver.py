from petsc4py.PETSc import TS, COMM_WORLD, DMPlex

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

    def initSolver(self, rhsFunction, convergedStepFunction, operators, fem):
        opers = { 'curl': operators.Curl, 'div': operators.DivSrT, 'srt': operators.SrT, 'fem': fem}
        self.setRHSFunction(rhsFunction, None , kargs=opers)
        self.setPostStep(fem.saveStep)
        self.setFromOptions()