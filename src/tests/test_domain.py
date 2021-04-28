import unittest
from domain.dmplex import BoxDM
from domain.elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

from functions.taylor_green import velocity_test, vorticity_test

class TestBoxDM(unittest.TestCase):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [5,5]}
    ngl = 3

    def setUp(self):
        self.dm_test = BoxDM()
        self.dm_test.create(self.test_case)
        self.dm_test.setFemIndexing(self.ngl)
        tot_nodes = 1
        for nel in self.test_case['nelem']:
            tot_nodes *= (self.ngl - 1) * (nel-1) + self.ngl

        self.tot_nodes = tot_nodes

    def test_create_vecs(self):
        vel, vort = self.dm_test.createVecs()

        assert vel.getName() == 'velocity'
        assert vort.getName() == 'vorticity'

        assert vel.getSize() == (self.tot_nodes * 2)
        assert vort.getSize() == self.tot_nodes

    def test_get_ngl(self):
        ngl_test = self.dm_test.getNGL() 
        assert ngl_test == self.ngl