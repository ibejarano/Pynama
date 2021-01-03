import unittest
import numpy as np
import yaml
from cases.immersed_boundary import ImmersedBoundaryStatic

class TestSearch(unittest.TestCase):
    # def setUp(self):
    #     # creates eulerian grid
    #     # creates a body with dl = h
    #     # it must
    #     # set malla 10x10 de 10 de largo y 10 alto
    #     # self.h = 1
    #     case = 'ibm-static'
    #     domain = {"nelem": [10,10] , "ngl": 3, "lower":[-5,-5] ,"upper":[5,5]}
    #     with open(f'src/cases/{case}.yaml') as f:
    #         yamlData = yaml.load(f, Loader=yaml.Loader)
    #     self.fem = ImmersedBoundaryStatic(yamlData, case=case, **domain)
    #     self.fem.setUp()

    caseYaml = 'ibm-static'
    caseOpts = {"nelem": [10,10] , "ngl": 3, "lower":[-5,-5] ,"upper":[5,5]}
    def setUp(self):
        with open(f'src/cases/{self.caseYaml}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        fem = ImmersedBoundaryStatic(yamlData, case=self.caseYaml, **self.caseOpts)
        fem.setUp()
        fem.setUpSolver()
        self.fem = fem

    def test_total_euler_nodes_finded(self):
        cells = self.fem.getAffectedCells(1)
        # send cells get nodes
        nodes, _ = self.fem.dom.getNodesCoordsFromEntities(cells)
        # ngl = 3
        assert len(nodes) == 5*5

        cells = self.fem.getAffectedCells(xSide=2, ySide=2)
        nodes, _ = self.fem.dom.getNodesCoordsFromEntities(cells)
        assert len(nodes) == 9*9

        cells = self.fem.getAffectedCells(xSide=1, ySide=2)
        nodes, _ = self.fem.dom.getNodesCoordsFromEntities(cells)
        
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

class TestDiracRegularGrid(unittest.TestCase):
    def setUp(self):
        case = 'ibm-static'
        domain = {"nelem": [10,10] , "ngl": 3, "lower":[-5,-5] ,"upper":[5,5]}
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        self.fem = ImmersedBoundaryStatic(yamlData, case=case, **domain)
        self.fem.setUp()
        self.fem.setUpSolver()

    def test_mass_conservation(self):
        D = self.fem.H
        _ , sizeGl = D.getSizes()[0]
        for i in range(sizeGl):
            mass = D.getRow(i)[1].sum()
            self.assertAlmostEqual(mass, 1, places=10)

    def test_momentum_conservation(self):
        D = self.fem.H
        _ , sizeGl = D.getSizes()[0]
        dim = self.fem.dom.getDimension()
        for lagInd in range(0, sizeGl, dim):
            eulerInd, diracs = D.getRow(lagInd)
            eulerNodes = eulerInd / dim
            lagNode = lagInd / dim
            eulerCoords = self.fem.dom.getNodesCoordinates(nodes=eulerNodes)
            lagCoord = self.fem.body.getNodeCoordinates(lagNode)
            dist = eulerCoords - lagCoord
            # print(dist)
            # print(dist[:,1])
            dist = dist[:,0] * dist[:,1]
            momentum = dist * diracs
            self.assertAlmostEqual(momentum.sum(), 0, places=10)

class TestDiracSpectralRegularGrid(unittest.TestCase):
    # TODO: Need to implement this
    def setUp(self):
        case = 'ibm-static'
        domain = {"nelem": [10,10] , "ngl": 5, "lower":[-5,-5] ,"upper":[5,5]}
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        self.fem = ImmersedBoundaryStatic(yamlData, case=case, **domain)
        self.fem.setUp()
        self.fem.setUpSolver()