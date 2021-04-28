import unittest
from domain.dmplex import BoxDM
from domain.elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

from functions.taylor_green import velocity_test, vorticity_test

class TestBoxDM(unittest.TestCase):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [8,5]}
    ngl = 5


    def setUp(self):
        self.dm_test = BoxDM()
        self.dm_test.create(self.test_case)
        self.dm_test.setFemIndexing(self.ngl)
        tot_nodes = 1
        for nel in self.test_case['nelem']:
            tot_nodes *= (self.ngl - 1) * (nel-1) + self.ngl

        self.tot_nodes = tot_nodes

        self.borders = [{'name':'up', 'faces': self.test_case['nelem'][0] },
        {'name':'down', 'faces': self.test_case['nelem'][0]}
        , {'name':'right', 'faces': self.test_case['nelem'][1]}
        , {'name':'left', 'faces': self.test_case['nelem'][1]}]
        for border in self.borders:
            faces = border['faces']
            border['dofs'] =  (self.ngl - 1) * (faces-1) + self.ngl

    def test_create_vecs(self):
        vel, vort = self.dm_test.createVecs()

        assert vel.getName() == 'velocity'
        assert vort.getName() == 'vorticity'

        assert vel.getSize() == (self.tot_nodes * 2)
        assert vort.getSize() == self.tot_nodes

    def test_get_ngl(self):
        ngl_test = self.dm_test.getNGL() 
        assert ngl_test == self.ngl

    def test_get_border_names(self):
        borderNames = self.dm_test.getBoundaryNames()
        assert len(borderNames) == 4
        for border in self.borders:
            assert border['name'] in borderNames

    def test_get_border_entities(self):
        for border in self.borders:
            faces = self.dm_test.getBorderEntities(border['name'])
            assert len(faces) == border['faces']

    def test_get_border_dofs(self):
        for border in self.borders:
            dofs = self.dm_test.getBorderDofs(border['name'])
            assert len(dofs) == (border['dofs'] * 2)

class TestBoxDMCaseB(TestBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [14,55]}
    ngl = 4
class TestBoxDMCaseC(TestBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [3,11]}
    ngl = 7

class TestBoxDMCaseD(TestBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [50,80]}
    ngl = 3