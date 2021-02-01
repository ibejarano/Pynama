from domain.dmplex import BoxDom
from domain.dmplex_bc import NewBoxDom
from domain.elements.spectral import Spectral
from viewer.paraviewer import Paraviewer
import unittest 
import numpy as np
import numpy.testing as np_test

class TestDMPlexNodes(unittest.TestCase):
    data = {"nelem": [2,2], "lower": [0,0], "upper":[1,1]}
    ngl = 4
    def setUp(self):
        dm = BoxDom()
        dm.create(self.data)
        dm.setFemIndexing(self.ngl)
        self.dm_ref = dm

        dmbc = NewBoxDom()
        dmbc.create(self.data)
        dmbc.setFemIndexing(self.ngl)
        self.dm_test = dmbc

        self.elem = Spectral(self.ngl, 2)

    # def test_cell_nodes(self):
    #     for cell in range(self.dm_ref.cellStart, self.dm_ref.cellEnd):
    #         print(cell)

    #         nodes = self.dm_ref.getGlobalNodesFromCell(cell, shared=True)
    #         inds = [node*2 + dof for node in nodes for dof in range(2)]

    #         print(inds)

    #         print("others", self.dm_test.getNodesFromCell(cell))
    #         exit()

    #     raise Exception('Not implemented yet')

    def test_coordinates(self):
        self.dm_test.computeFullCoordinates(self.elem, ngl=self.ngl)
        self.dm_ref.computeFullCoordinates(self.elem)
        viewer = Paraviewer()
        viewer.configure(2)


        # viewer.saveMesh(self.dm_ref.fullCoordVec, 'meshref')
        viewer.saveMesh(self.dm_test.fullCoordVec)

        arr_ref = self.dm_ref.fullCoordVec
        arr_test = self.dm_test.fullCoordVec

        np_test.assert_array_almost_equal(arr_ref, arr_test, decimal=10)
        # viewer.saveData(0, 0.0,self.dm_ref.fullCoordVec)
        # viewer.writeXmf('probando')