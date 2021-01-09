import unittest
from domain.dmplex import BoxDom, GmshDom
from domain.elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc


class TestBoxDMPLEX2D(unittest.TestCase):

    def setUp(self):
        ngl = 3
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [3,4]}
        self.dom = BoxDom()
        self.dom.create(data2D)
        self.dom.setFemIndexing(ngl)

    def test_cell_start_end(self):
        self.assertEqual(self.dom.cellStart, 0)
        self.assertEqual(self.dom.cellEnd, 12)

    def test_cell_corners_coords(self):
        coords_cell_0 = np.array([[0,0 ],[0.2,0],[0.2,0.2],[0,0.2]])
        coords_cell_0.shape = 8
        coord= self.dom.getCellCornersCoords(0)
        np_test.assert_array_almost_equal(coords_cell_0, coord, decimal=13)

    def test_borders_nodes(self):
        total = 28
        bordersNodes = self.dom.getBordersNodes()
        bordersNodes_alt = self.dom.getNodesFromLabel("External Boundary")
        assert type(bordersNodes) == set
        assert len(bordersNodes) == total

        assert type(bordersNodes_alt) == set
        assert len(bordersNodes_alt) == total

        np_test.assert_equal(bordersNodes, bordersNodes_alt)

    def test_border_nodes(self):
        borderNames = self.dom.getBordersNames()
        for b in borderNames:
            if b in ['up', 'down']:
                assert len(self.dom.getBorderNodes(b)) == 7
            else:
                assert len(self.dom.getBorderNodes(b)) == 9

    def test_edge_width(self):
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [4,4]}
        dm = BoxDom()
        dm.create(data2D)
        dm.setFemIndexing(3)

        width_test = dm.getEdgesWidth()

        width_ref = (data2D['upper'][0] - data2D['lower'][0] )/ data2D['nelem'][0]

        assert width_ref == width_test

class TestNglIndexing2D(unittest.TestCase):
    def setUp(self):
        self.doms = list()
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [2,3]}
        for ngl in range(2, 10 , 2):
            dm = BoxDom()
            dm.create(data2D)
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

class TestBoxDMPLEX3D(unittest.TestCase):

    def setUp(self):
        data3D = {'lower': [0,0,0] , 'upper':[0.6,0.8,1], "nelem": [3,4,5]}
        self.dom = BoxDom()
        self.dom.create(data3D)
        self.dom.setFemIndexing(3)

    def test_generate_dmplex(self):
        assert self.dom.getDimension() == 3

    def test_cell_start_end(self):
        self.assertEqual(self.dom.cellStart, 0)
        self.assertEqual(self.dom.cellEnd, 3*4*5)

    def test_cell_corners_coords(self):
        coords_cell_0 = np.array(
            [[0,0,0 ] , [0,0.2,0],
            [0.2,0.2,0], [0.2,0,0],
            [0,0,0.2 ],[0.2,0,0.2],
            [0.2,0.2,0.2],
            [0,0.2,0.2]
        ])
        coords_cell_0.shape = 8*3
        coord= self.dom.getCellCornersCoords(0)
        np_test.assert_array_almost_equal(coords_cell_0, coord, decimal=13)


    def test_border_names(self):
        borderNames = self.dom.getBordersNames()

        assert len(borderNames) == 6
        for b in borderNames:
            assert b in ['up', 'down', 'left', 'right', 'front', 'back']

    def test_borders_nodes(self):
        totalArc = 28*11
        totalFace = 35*2
        total = totalArc + totalFace
        bordersNodes = self.dom.getBordersNodes()
        bordersNodes_alt = self.dom.getNodesFromLabel("External Boundary")
        assert type(bordersNodes) == set
        assert len(bordersNodes) == total

        assert type(bordersNodes_alt) == set
        assert len(bordersNodes_alt) == total

        np_test.assert_equal(bordersNodes, bordersNodes_alt)

    def test_border_nodes(self):
        borderNames = self.dom.getBordersNames()
        for b in borderNames:
            if b in ['up', 'down']:
                assert len(self.dom.getBorderNodes(b)) == 7*11
            elif b in ['left', 'right']:
                assert len(self.dom.getBorderNodes(b)) == 9*11
            else:
                assert len(self.dom.getBorderNodes(b)) == 7*9

    def test_edge_width(self):
        data3D = {'lower': [0,0,0] , 'upper':[0.8,0.8,0.8], "nelem": [4,4,4]}
        dm = BoxDom()
        dm.create(data3D)
        dm.setFemIndexing(3)

        width_test = dm.getEdgesWidth()

        width_ref = (data3D['upper'][0] - data3D['lower'][0] )/ data3D['nelem'][0]

        assert width_ref == width_test

class TestNglIndexing3D(unittest.TestCase):
    def setUp(self):
        self.doms = list()
        data3D = {'lower': [0,0,0] , 'upper':[0.6,0.8,1], "nelem": [2,3,4]}
        for ngl in range(2, 10 , 2):
            dm = BoxDom()
            dm.create(data3D)
            dm.setFemIndexing(ngl)
            self.doms.append(dm)

    def test_borders_nodes_num(self):
        edges = 36 + 68
        cells = 52
        cornerNodes = 54
        for dom in self.doms:
            ngl = dom.getNGL()
            total = cornerNodes + edges*(ngl-2) + cells*((ngl-2)**2)
            assert len(dom.getBordersNodes()) == total

    def test_border_nodes_num(self):
        borderNames = self.doms[0].getBordersNames()
        for dom in self.doms:
            ngl = dom.getNGL()
            for b in borderNames:
                if b in ['up', 'down']:
                    total = 15 + 22*(ngl-2) + 8*((ngl-2)**2)
                    assert len(dom.getBorderNodes(b)) == total
                elif b in ['right', 'left']:
                    total = 20 + 31*(ngl-2) + 12*((ngl-2)**2)
                    assert len(dom.getBorderNodes(b)) == total
                elif b in ['front', 'back']:
                    total = 12 + 17*(ngl-2) + 6*((ngl-2)**2)
                    assert len(dom.getBorderNodes(b)) == total
                else:
                    raise Exception("Not found Border")

class DomainModTests2D(unittest.TestCase):

    def setUp(self):
        ngl = 2
        data2D = {'lower': [0,0] , 'upper':[1,1], "nelem": [2,2]}
        self.dom = BoxDom()
        self.dom.create(data2D)
        self.dom.setFemIndexing(ngl)

        dim = self.dom.getDimension()
        spectral2D = Spectral(ngl,dim)
        self.dom.computeFullCoordinates(spectral2D)

        self.testVelVec = self.dom.createGlobalVec()

    def test_get_all_global_nodes_ngls(self):
        data2D = {'lower': [0,0] , 'upper':[0.6,0.8], "nelem": [2,3]}
        for ngl in range(2, 14):
            dom = BoxDom()
            dom.create(data2D)
            dom.setFemIndexing(ngl)
            allNodes = dom.getAllNodes()
            total = 12 + 17*(ngl-2) + 6*((ngl-2)**2)
            assert len(allNodes) == total
            del dom

    def test_get_nodes_coordinates_2D(self):
        allNodes = self.dom.getAllNodes()
        coords = [[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]]
        test_coords = self.dom.getNodesCoordinates(allNodes)
        np_test.assert_array_almost_equal(coords, test_coords)

#     # TODO: implement 3D tests

    def test_set_function_vec_to_vec_2D(self):
        np_coords = np.array([[0., 0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5], [0.5, 0.5], [1.,  0.5], [0.,  1. ], [0.5, 1. ], [1.,  1. ]])
        result = np.sqrt(np_coords)
        allNodes = self.dom.getAllNodes()

        f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))
        dim = self.dom.getDimension()
        self.dom.applyFunctionVecToVec(allNodes, f, self.testVelVec,dim)
        test_result = self.testVelVec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_function_vec_to_vec_2D_some_nodes(self):
        np_coords = np.array([[0., 0. ], [1, 1. ], [1.,  1. ], [0.,  0.5], [1., 1.], [1.,  1.], [1.,  1. ], [0.5, 1. ], [1.,  1. ]])
        result = np.sqrt(np_coords)

        self.testVelVec.set(1.0)
        someNodes = [0,3,7] 
        f = lambda coords : (sqrt(coords[0]),sqrt(coords[1]))

        dim = self.dom.getDimension()
        self.dom.applyFunctionVecToVec(someNodes, f, self.testVelVec, dim)
        test_result = self.testVelVec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_function_scalar_to_vec_2D(self):
        vecScalar = PETSc.Vec().createSeq(9)

        result = np.array([0,0.5,1,0.5,1.,1.5,1,1.5,2])

        allNodes = self.dom.getAllNodes()
        f = lambda coord: (coord[0]+coord[1])
        self.dom.applyFunctionScalarToVec(allNodes, f , vecScalar)
        test_result = vecScalar.getArray()
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    def test_set_constant_to_vec_2D(self):

        cteValue = [3 , 5]

        result = np.array( [3,5]* 9).reshape(9, 2)
        vec = PETSc.Vec().createSeq(18)
        self.dom.applyValuesToVec(self.dom.getAllNodes(), cteValue, vec)
        test_result = vec.getArray().reshape(9,2)
        np_test.assert_array_almost_equal(result, test_result, decimal=12)

    # def test_set_function_to_vec(self):
    #     raise NotImplementedError