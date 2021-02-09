import unittest 
from solver.ts_solver import TSSolver
import numpy as np
import numpy.testing as np_test
from petsc4py import PETSc


from matrices.operators import Operators
from functions.taylor_green import velocity, vorticity, alpha

from main import MainProblem
from domain.elements.spectral import Spectral


class TestTSSetup(unittest.TestCase):
    nu = 0.5/0.01
    nelem = [40,40]
    ngl = 3
    domain = {'nelem': nelem, 'lower': [0, 0], 'upper': [1, 1]}

    def setUp(self):
        fem  = MainProblem('taylor-green')
        fem.setUp()
        self.fem = fem
        operators = Operators()
        self.operators = operators


        dm = fem.dm.velDM
        self.operators.preallocate(self.domain, self.ngl)
        self.operators.setDM(dm)
        dim = dm.getDimension()
        elem = Spectral(self.ngl, dim)
        self.operators.setElem(elem)
        self.operators.assemble()

        ts = TSSolver()
        ts.setUpTimes(0, 1.0, 100)
        ts.setDM(self.fem.dm.vortDM)

        ts.initSolver(rhsFunc, operators, fem)
        self.ts = ts

    def test_func_eval(self):
        dm = self.ts.getDM()
        self.fem.computeInitialConditions()
        globalVort = dm.getGlobalVec()
        localVort = dm.getLocalVec()
        dm.localToGlobal(localVort, globalVort)
        dm.restoreLocalVec(localVort)
        self.ts.solve(globalVort)
        raise Exception('noera')

