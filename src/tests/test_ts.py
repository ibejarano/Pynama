import unittest 
from solver.ts_solver import TsSolver
import numpy as np
import numpy.testing as np_test
from petsc4py import PETSc

class TaylorGreenTest(unittest.TestCase):
    def setUp(self):
        self.ts = TsSolver()

    def test_rungeKutta(self):
        assert self.ts.getType() == 'rk'
        assert self.ts.getRKType() == '5bs'

    def test_timesSetUp(self):
        self.ts.setUpTimes(1.2 , 3.4, 50)

        assert self.ts.getTime() == 1.2
        assert self.ts.getMaxTime() == 3.4
        assert self.ts.getMaxSteps() == 50