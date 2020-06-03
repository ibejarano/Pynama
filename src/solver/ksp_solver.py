from petsc4py.PETSc import KSP

class KspSolver(KSP):
    def __init__(self):
        pass

    def createSolver(self, mat, comm):
        # self.logger.debug("setupKSP")
        # create linear solver
        # ksp = PETSc.KSP()
        self.create(comm)
        self.setFromOptions()
        self.setOperators(mat)
        self.setUp()
