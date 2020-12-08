from petsc4py.PETSc import KSP
from petsc4py.PETSc import PC

class KspSolver(KSP):
    def __init__(self):
        pass

    def createSolver(self, mat, comm):
        # self.logger.debug("setupKSP")
        # create linear solver
        # ksp = PETSc.KSP()
        self.create(comm)
        self.setType('preonly')
        pc = PC().create()
        pc.setType('lu')
        self.setPC(pc)
        self.setFromOptions()
        self.setOperators(mat)
        self.setUp()
