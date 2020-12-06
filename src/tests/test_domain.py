import unittest
from domain.dmplex import DMPlexDom
from elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

class TestBoxDMPLEX2D(unittest.TestCase):

    # Test Roadmap
    ## 1. Separate dmplex from indicesmanager
    ## 2. Test the exact numer of the nodes in the border

    def setUp(self):
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [3,4]}
        self.dom = DMPlexDom(boxMesh=data2D)
        self.dom.setFemIndexing(3)

    def test_generate_dmplex(self):
        assert self.dom.getDimension() == 2

    def test_cell_start_end(self):
        self.assertEqual(self.dom.cellStart, 0)
        self.assertEqual(self.dom.cellEnd, 12)

    def test_cell_corners_coords(self):
        coords_cell_0 = np.array([[0,0 ],[0.2,0],[0.2,0.2],[0,0.2]])
        coords_cell_0.shape = 8
        coord= self.dom.getCellCornersCoords(0)
        np_test.assert_array_almost_equal(coords_cell_0, coord, decimal=13)

    def test_borders_nodes(self):
        # total is for 3,4 ngl=3 box mesh
        # Cambiar por exact num, dejar la cantidad para ngl variable
        total = 28
        assert len(self.dom.getBordersNodes()) == total

    def test_border_nodes(self):
        borderNames = self.dom.getBordersNames()
        for b in borderNames:
            if b in ['up', 'down']:
                assert len(self.dom.getBorderNodes(b)) == 7
            else:
                assert len(self.dom.getBorderNodes(b)) == 9
        

class TestNglIndexing2D(unittest.TestCase):
    def setUp(self):
        self.doms = list()
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [2,3]}
        for ngl in range(2, 10 , 2):
            dm = DMPlexDom(boxMesh=data2D)
            dm.setFemIndexing(ngl)
            self.doms.append(dm)

    def test_borders_nodes_num(self):
        cornerNodes = 10
        for dom in self.doms:
            ngl = dom.getNGL()
            total = cornerNodes + 10*(ngl-2)
            assert len(dom.getBordersNodes()) == total

    def test_border_nodes_num(self):
        borderNames = self.doms[0].getBordersNames()
        for dom in self.doms:
            ngl = dom.getNGL()
            for b in borderNames:
                if b in ['up', 'down']:
                    total = 3 + 2*(ngl-2)
                    assert len(dom.getBorderNodes(b)) == total
                else:
                    total = 4 + 3*(ngl-2)
                    assert len(dom.getBorderNodes(b)) == total

# class TestBoxDMPLEX3D(unittest.TestCase):
#     def setUp(self):
#         data2D = {}
#         self.dom_list_2d = list()
#         dim = 2
#         for ngl in range(2,4):
#             dm = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
#             dm.setFemIndexing(ngl)
#             self.dom_list_2d.append(dm)

#         self.dom_list_3d = list()

#         dim = 3
#         for ngl in range(2,4):
#             dm = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
#             dm.setFemIndexing(ngl)

#             self.dom_list_3d.append(dm)

#     def test_generate_dmplex(self):
#         for dom in self.dom_list_3d:
#             self.assertEqual(dom.getDimension(), 3)

#     def test_cell_start_end(self):
#         for dom in self.dom_list_3d:
#             self.assertEqual(dom.cellStart, 0)
#             self.assertEqual(dom.cellEnd, 8)

#     def test_cell_corners_coordshape(self):
#         shape = 8*3
#         for dom in self.dom_list_3d:
#             coords = dom.getCellCornersCoords(0)
#             self.assertEqual(coords.shape[0], shape)

# class DomainModTests(unittest.TestCase):

#     def setUp(self):
#         dim = 2
#         self.dom2d = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
#         self.dom2d.setFemIndexing(2)
#         spectral2D = Spectral(2,2)
#         self.dom2d.computeFullCoordinates(spectral2D)

#         self.testVelVec = self.dom2d.createGlobalVec()
#         self.dim = dim

#     def test_get_all_global_nodes_2D(self):
#         allNodes = self.dom2d.getAllNodes()
#         self.assertEqual(len(allNodes), 9)

#     def test_get_all_global_nodes_2D_ngls(self):
#         dim = 2
#         ngl = 3
#         dom = DMPlexDom([0]*dim, [1]*dim, [2]*dim)
#         dom.setFemIndexing(ngl)
#         allNodes = dom.getAllNodes()
#         self.assertEqual(len(allNodes), 25)

#     def test_get_nodes_coordinates_2D(self):
#         allNodes = self.dom2d.getAllNodes()
#         coords = [[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]]
#         test_coords = self.dom2d.getNodesCoordinates(allNodes)
#         np_test.assert_array_almost_equal(coords, test_coords)

#     # TODO: implement 3D tests

#     def test_set_function_vec_to_vec_2D(self):
#         np_coords = np.array([[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]])
#         result = np.sqrt(np_coords)
#         allNodes = self.dom2d.getAllNodes()
#         f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))
#         self.dom2d.applyFunctionVecToVec(allNodes, f, self.testVelVec,self.dim)
#         test_result = self.testVelVec.getArray().reshape(9,2)
#         np_test.assert_array_almost_equal(result, test_result, decimal=12)

#     def test_set_function_vec_to_vec_2D_some_nodes(self):
#         np_coords = np.array([[0., 0. ], [0, 0. ], [0.,  0. ], [0.,  0.5], [0., 0.], [0.,  0.], [0.,  0. ], [0.5, 1. ], [0.,  0. ]])
#         result = np.sqrt(np_coords)
#         self.testVelVec.set(0.0)
#         someNodes = [0,3,7] 
#         f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))
#         self.dom2d.applyFunctionVecToVec(someNodes, f, self.testVelVec,self.dim)
#         test_result = self.testVelVec.getArray().reshape(9,2)
#         np_test.assert_array_almost_equal(result, test_result, decimal=12)

#     def test_set_function_scalar_to_vec_2D(self):
#         vecScalar = PETSc.Vec().createSeq(9)
#         result = np.array([0,0.5,1,0.5,1.,1.5,1,1.5,2])
#         allNodes = self.dom2d.getAllNodes()
#         f = lambda coord: (coord[0]+coord[1])
#         self.dom2d.applyFunctionScalarToVec(allNodes, f , vecScalar)
#         test_result = vecScalar.getArray()
#         np_test.assert_array_almost_equal(result, test_result, decimal=12)

#     def test_set_constant_to_vec_2D(self):
#         result = np.array( [3,5]* 9).reshape(9, 2)
#         vec = PETSc.Vec().createSeq(18)
#         self.dom2d.applyValuesToVec(self.dom2d.getAllNodes(), [3,5], vec)
#         test_result = vec.getArray().reshape(9,2)
#         np_test.assert_array_almost_equal(result, test_result, decimal=12)

    # def test_set_function_to_vec(self):
    #     raise NotImplementedError