import unittest
from matrices.operators import Operators
from functions.taylor_green import velocity, vorticity, alpha
import numpy as np
import numpy.testing as np_test
from main import MainProblem
from solver.ts_solver import TSSolver
from domain.elements.spectral import Spectral

class TestKrhs(unittest.TestCase):
    nu = 0.5/0.01
    nelem = [2,2]
    ngl = 3
    domain = {'nelem': nelem, 'lower': [0, 0], 'upper': [1, 1]}
    def setUp(self):
        fem  = MainProblem('taylor-green')
        fem.setUp()
        self.fem = fem

    def test_preallocation(self):
        operators = Operators()
        operators.preallocate(self.domain, self.ngl)

        curl= operators.Curl
        div = operators.DivSrT
        srt = operators.SrT

        assert curl.getSize() == (25, 50)
        assert div.getSize() == (50, 75)
        assert srt.getSize() == (75, 50)

    def test_vorticity_vec(self):
        operators = Operators()
        operators.preallocate(self.domain, self.ngl)
        vort = operators.Curl.createVecLeft()
        assert vort.getSize() == 25
        vort.assemble()

    def test_assemble_mats(self):
        dm = self.fem.dm.velDM
        ngl = self.fem.dm.getNGL()
        operators = Operators()
        operators.preallocate(config=self.domain, ngl=ngl)
        operators.setDM(dm)
        operators.setElem(self.fem.elem)
        operators.assemble()