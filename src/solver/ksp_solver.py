from petsc4py.PETSc import KSP, PC, COMM_WORLD
from petsc4py.PETSc import PC
import logging


class KleSolver:
    def __init__(self):
        self.logger = logging.getLogger("KLE Solver")
    
    def setMat(self, mat):
        self.mat = mat

    def setUp(self):
        solveType = self.mat.bcType
        K = self.mat.K
        self.solver = KspSolver()
        self.solver.createSolver(K)

        self.__vel = K.createVecRight()
        self.__vel.setName("velocity")
        self.__isNS = False

        if solveType == "NS":
            Kfs = self.mat.Kfs
            self.solverFS = KspSolver()
            self.solverFS.createSolver(K + Kfs)
            self.__velFS = K.createVecRight()
            self.__isNS = True
            
    def isNS(self):
        return self.__isNS

    def solve(self, vort):
        self.solver(self.mat.Rw * vort + self.mat.Krhs * self.__vel, self.__vel)

    def solveFS(self, vort):
        self.solverFS( self.mat.Rw * vort + self.mat.Rwfs * vort\
             + self.mat.Krhsfs * self.__vel , self.__velFS)

    def getFreeSlipSolution(self):
        return self.__velFS

    def getSolution(self):
        return self.__vel

class KspSolver(KSP):
    comm = COMM_WORLD
    def __init__(self):
        self.logger = None

    def createSolver(self, mat):
        self.logger = logging.getLogger("KSP Solver")
        self.logger.debug("setupKSP")
        self.create(self.comm)
        self.setType('preonly')
        pc = PC().create()
        pc.setType('lu')
        self.setPC(pc)
        self.setFromOptions()
        self.setOperators(mat)
        self.setUp()