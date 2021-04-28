import unittest
from domain.dmplex import BoxDM
from domain.elements.spectral import Spectral
import numpy as np

class TestCoordBoxDM(unittest.TestCase):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [3,3]}
    ngl = 3

    def setUp(self):
        self.dm_test = BoxDM()
        self.dm_test.create(self.test_case)
        self.dm_test.setFemIndexing(self.ngl)
        dim = self.dm_test.getDimension()
        self.sp_elem = Spectral(self.ngl, dim)
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

    def test_create_full_coord_vec(self):

        coord_vec = self.dm_test.computeFullCoordinates(self.sp_elem)
        dim = self.dm_test.getDimension()
        # Test correct size
        assert coord_vec.getSize() == (self.tot_nodes * dim)

        # Test un-repeated coords
        coord_arr = coord_vec.getArray().reshape((self.tot_nodes, dim))
        coord_unique = np.unique(coord_arr, axis=0)
        assert len(coord_unique) == len(coord_arr)

class TestCoordBoxDMCaseA(TestCoordBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [3,3]}
    ngl = 5

class TestCoordBoxDMCaseB(TestCoordBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [5,9]}
    ngl = 9

class TestCoordBoxDMCaseC(TestCoordBoxDM):
    test_case = {'lower': [0,0], 'upper': [1,1], 'nelem': [50,65]}
    ngl = 3