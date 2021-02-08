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
        operators = Operators()
        self.operators = operators

    def test_preallocation(self):
        self.operators.preallocate(self.domain, self.ngl)

        curl = self.operators.Curl
        div = self.operators.DivSrT
        srt = self.operators.SrT

        assert curl.getSize() == (25, 50)
        assert div.getSize() == (50, 75)
        assert srt.getSize() == (75, 50)

    def test_vorticity_vec(self):
        self.operators.preallocate(self.domain, self.ngl)
        vort = self.operators.Curl.createVecLeft()
        assert vort.getSize() == 25
        vort.assemble()

    def test_assemble_mats(self):
        dm = self.fem.dm.velDM
        self.operators.preallocate(self.domain, self.ngl)
        self.operators.setDM(dm)
        dim = dm.getDimension()
        elem = Spectral(self.ngl, dim)
        self.operators.setElem(elem)
        self.operators.assemble()