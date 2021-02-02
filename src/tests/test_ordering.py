import unittest
from matrices.new_mat import Matrices
from domain.dmplex_bc import NewBoxDom
from solver.kle_solver import KspSolver
from domain.elements.spectral import Spectral
from functions.taylor_green import velocity, vorticity, alpha
import numpy as np
import numpy.testing as np_test

import yaml
from cases.base_problem import BaseProblem 


class TestOrdering(unittest.TestCase):

    def setUp(self):
        domain = {'nelem': [3, 3], 'lower': [0, 0], 'upper': [1, 1]}
        ngl = 4


        dm = NewBoxDom()
        dm.create(domain)
        dm.setFemIndexing(ngl)
        dm.createElement()
        dm.computeFullCoordinates()
        mats = Matrices()
        mats.setDM(dm)

        self.dm_test = dm

        with open('src/cases/taylor-green.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)

        # self.d = yamlData
        fem_ref = BaseProblem(yamlData)
        fem_ref.setUpDomain()
        fem_ref.readMaterialData()
        fem_ref.setUpSolver()
        fem_ref.dom.computeFullCoordinates()
        # fem_ref.setUpInitialConditions()

        self.fem_ref = fem_ref

    def test_get_vel_dofs(self):
        dm_test = self.dm_test
        dm_ref = self.fem_ref.dom.getDM()
        borderCells = dm_test.getStratumIS('boundary', 1)
        assert dm_test.getDimension() == 2, "Check if celltype = 4 in dim = 3"
        allCells = dm_test.getStratumIS('celltype', 4)
        insideCells = allCells.difference(borderCells)
        lgmap = dm_test.getLGMap()

        for cell in insideCells.getIndices():
            test_dofs = dm_test.getLocalVelocityDofsFromCell(cell)
            ref_nodes = dm_ref.getGlobalNodesFromCell(cell, True)

            ref_dofs = dm_ref.getVelocityIndex(ref_nodes)

            np_test.assert_array_equal(test_dofs, ref_dofs)