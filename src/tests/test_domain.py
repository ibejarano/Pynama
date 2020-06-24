import unittest
from domain.dmplex import DMPlexDom
from elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

class DomiainTest(unittest.TestCase):
    def setUp(self):
        self.dom_list_2d = list()
        dim = 2
        for ngl in range(2,4):
            dm = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
            dm.setFemIndexing(ngl)
            self.dom_list_2d.append(dm)

        self.dom_list_3d = list()

        dim = 3
        for ngl in range(2,4):
            dm = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
            dm.setFemIndexing(ngl)

            self.dom_list_3d.append(dm)


    def test_generate_dmplex2d(self):
        for dom in self.dom_list_2d:
            self.assertEqual(dom.getDimension(), 2)

    def test_generate_dmplex3d(self):
        for dom in self.dom_list_3d:
            self.assertEqual(dom.getDimension(), 3)

    def test_cell_start_end_2d(self):
        for dom in self.dom_list_2d:
            self.assertEqual(dom.cellStart, 0)
            self.assertEqual(dom.cellEnd, 4)

    def test_cell_start_end_3d(self):
        for dom in self.dom_list_3d:
            self.assertEqual(dom.cellStart, 0)
            self.assertEqual(dom.cellEnd, 8)
    
    def test_cell_corners_coords_2d(self):
        coords_cell_0 = np.array([[0,0 ],[0.5,0],[0.5,0.5],[0,0.5]])
        coords_cell_0.shape = 8
        for dom in self.dom_list_2d:
            coord=dom.getCellCornersCoords(0)
            np_test.assert_array_almost_equal(coords_cell_0, coord, decimal=10)
            # print(coord)

    def test_cell_corners_coordsshape_3d(self):
        shape = 8*3
        for dom in self.dom_list_3d:
            coords = dom.getCellCornersCoords(0)
            self.assertEqual(coords.shape[0], shape)


class DomainModTests(unittest.TestCase):

    def setUp(self):
        dim = 2
        self.dom2d = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
        self.dom2d.setFemIndexing(2)
        spectral2D = Spectral(2,2)
        self.dom2d.computeFullCoordinates(spectral2D)

        self.testVelVec = self.dom2d.createGlobalVec()
        self.dim = dim

    def test_get_all_global_nodes_2D(self):
        allNodes = self.dom2d.getAllNodes()
        self.assertEqual(len(allNodes), 9)

    def test_get_all_global_nodes_2D_ngls(self):
        dim = 2
        ngl = 3
        dom = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
        dom.setFemIndexing(ngl)
        allNodes = dom.getAllNodes()
        self.assertEqual(len(allNodes), 25)

    def test_get_nodes_coordinates_2D(self):
        allNodes = self.dom2d.getAllNodes()
        coords = [[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]]
        test_coords = self.dom2d.getNodesCoordinates(allNodes)
        np_test.assert_array_almost_equal(coords, test_coords)

    # TODO: implement 3D tests

    def test_set_function_vec_to_vec_2D(self):
        np_coords = np.array([[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]])
        result = np.sqrt(np_coords)
        allNodes = self.dom2d.getAllNodes()
        f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))
        self.dom2d.applyFunctionVecToVec(allNodes, f, self.testVelVec,self.dim)
        test_result = self.testVelVec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_function_vec_to_vec_2D_some_nodes(self):
        np_coords = np.array([[0., 0. ], [0, 0. ], [0.,  0. ], [0.,  0.5], [0., 0.], [0.,  0.], [0.,  0. ], [0.5, 1. ], [0.,  0. ]])
        result = np.sqrt(np_coords)
        self.testVelVec.set(0.0)
        someNodes = [0,3,7] 
        f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))
        self.dom2d.applyFunctionVecToVec(someNodes, f, self.testVelVec,self.dim)
        test_result = self.testVelVec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_function_scalar_to_vec_2D(self):
        vecScalar = PETSc.Vec().createSeq(9)
        result = np.array([0,0.5,1,0.5,1.,1.5,1,1.5,2])
        allNodes = self.dom2d.getAllNodes()
        f = lambda coord: (coord[0]+coord[1])
        self.dom2d.applyFunctionScalarToVec(allNodes, f , vecScalar)
        test_result = vecScalar.getArray()
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_constant_to_vec_2D(self):
        result = np.array( [3,5]* 9).reshape(9, 2)
        vec = PETSc.Vec().createSeq(18)
        self.dom2d.applyValuesToVec(self.dom2d.getAllNodes(), [3,5], vec)
        test_result = vec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    # def test_set_function_to_vec(self):
    #     raise NotImplementedError