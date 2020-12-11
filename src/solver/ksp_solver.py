from petsc4py.PETSc import KSP
from petsc4py.PETSc import PC
import logging

class KspSolver(KSP):
    def __init__(self):
        self.logger = None

    def createSolver(self, mat, comm):
        self.logger = logging.getLogger("KSP Solver")
        self.logger.debug("setupKSP")
        self.create(comm)
        self.setType('preonly')
        pc = PC().create()
        pc.setType('lu')
        self.setPC(pc)
        self.setFromOptions()
        self.setOperators(mat)
        self.setUp()