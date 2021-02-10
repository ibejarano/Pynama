from petsc4py.PETSc import KSP, PC, COMM_WORLD
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
            self.__velFS.setName('free-slip')
            self.__isNS = True
            
    def isNS(self):
        return self.__isNS

    def solve(self, vort, vec=None):
        if not vec:
            self.solver(self.mat.Rw * vort + self.mat.Krhs * self.__vel, self.__vel)
        else:
            self.solver(self.mat.Rw * vort + self.mat.Krhs * vec, vec)

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
        self.setType('gmres')
        pc = PC().create()
        pc.setType('lu')
        self.setPC(pc)
        self.setFromOptions()
        self.setOperators(mat)
        self.setUp()