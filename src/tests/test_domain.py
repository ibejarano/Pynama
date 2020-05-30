import unittest
from domain.dmplex import DMPlexDom
import numpy as np
import numpy.testing as np_test

class DomiainTest(unittest.TestCase):
    def setUp(self):
        self.dom_list_2d = list()
        dim = 2
        for ngl in range(2,4):
            dm = DMPlexDom(dim)
            dm.setUpDmPlex([0]*dim, [1]*dim, [2]*dim)
            dm.setFemIndexing(ngl)

            self.dom_list_2d.append(dm)

        self.dom_list_3d = list()

        dim = 3
        for ngl in range(2,4):
            dm = DMPlexDom(dim)
            dm.setUpDmPlex([0]*dim, [1]*dim, [2]*dim)
            dm.setFemIndexing(ngl)

            self.dom_list_3d.append(dm)


    def test_generate_dmplex2d(self):
        for dom in self.dom_list_2d:
            self.assertEqual(dom.dm.getDimension(), 2)

    def test_generate_dmplex3d(self):
        for dom in self.dom_list_3d:
            self.assertEqual(dom.dm.getDimension(), 3)

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

