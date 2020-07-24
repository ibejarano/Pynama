import unittest
import numpy as np
from cases.immersed_boundary import ImmersedBoundaryStatic

class TestSearch(unittest.TestCase):
    def setUp(self):
        # creates eulerian grid
        # creates a body with dl = h
        # it must
        # set malla 10x10 de 10 de largo y 10 alto
        # self.h = 1
        self.fem = ImmersedBoundaryStatic(case='ibm-static')
        self.fem.setUp()

    def test_total_euler_nodes_finded(self):
        cells = self.fem.getAffectedCells(1)
        # send cells get nodes
        nodes = self.fem.dom.getGlobalNodesFromEntities(cells, shared=False)
        # ngl = 3
        assert len(nodes) == 5*5

        cells = self.fem.getAffectedCells(xSide=2, ySide=2)
        nodes = self.fem.dom.getGlobalNodesFromEntities(cells, shared=False)
        assert len(nodes) == 9*9

        cells = self.fem.getAffectedCells(xSide=1, ySide=2)
        nodes = self.fem.dom.getGlobalNodesFromEntities(cells, shared=False)
        
        assert len(nodes) == 5*9

    def test_affected_cells_center_origin(self):
        cells = self.fem.getAffectedCells(1)
        assert len(cells)== 2 * 2

        cells = self.fem.getAffectedCells(xSide=2, ySide=2)
        assert len(cells)== 4 * 4

        cells = self.fem.getAffectedCells(xSide=1, ySide=2)
        assert len(cells)== 2 * 4

    def test_affected_cells_center_offset(self):
        center = np.array([0.5, 0.5])
        cells = self.fem.getAffectedCells(1, center=center)
        assert len(cells)== 1

        cells = self.fem.getAffectedCells(xSide=2, ySide=2, center=center)
        assert len(cells)== 3 * 3

        cells = self.fem.getAffectedCells(xSide=1, ySide=2, center=center)
        assert len(cells)== 1 * 3

class TestDirac(unittest.TestCase):
    def setUp(self):
        self.ibm = ImmersedBoundaryStatic(case='ibm-static')
        self.ibm.setUp()
        self.ibm.setUpSolver()

    def test_mass_conservation(self):
        assert 2==1

    def test_momentum_conservation(self):
        pass