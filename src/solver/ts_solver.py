from petsc4py.PETSc import TS

class TsSolver(TS):
    def __init__(self, comm):
        self.create(comm=comm)
        self.setProblemType(self.ProblemType.NONLINEAR)  # Should we use LINEAR?
        self.setEquationType(self.EquationType.ODE_EXPLICIT)
        self.setType(self.Type.RK)
        self.setRKType(self.RKType.RK5F)

    def setUpTimes(self, sTime, eTime, steps):
        self.setTime(sTime)
        # ts.setTimeStep(1e-6)
        # ts.setInitialTimeStep(sTime, (eTime - sTime) / (100 * maxSteps))
        # ts.setDuration(eTime, max_steps=maxSteps)
        self.setMaxTime(eTime)
        self.setMaxSteps(steps)
        self.setExactFinalTime(self.ExactFinalTime.MATCHSTEP)
        #ts.setExactFinalTime(ts.ExactFinalTime.INTERPOLATE) N
        # Sundials doesn't support MATCHSTEP (-ts_exact_final_time INTERPOLATE)

    def initSolver(self, rhsFunction, convergedStepFunction):
        # init variables for solution
        # self.initKLErhsFunction()
        # Set rhs function
        self.setRHSFunction(rhsFunction)
        # ts.setIFunction(self.evalKLEiFunction, self.f)
        # ts.setIJacobian(self.evalKLEiJacobian, self.Jac)

        # Set fuction to be run after each step
        self.setPostStep(convergedStepFunction)
        self.setFromOptions()