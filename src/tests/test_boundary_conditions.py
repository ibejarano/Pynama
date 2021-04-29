from utils.yaml_handler import readYaml
from domain.dmplex import BoxDM

import unittest
import numpy as np
import numpy.testing as np_test

from domain.boundaries.boundary_conditions import BoundaryConditions

class TestBoundaryConditions(unittest.TestCase):
    bcNames = ['up','down', 'right', 'left']

    def setUp(self):
        config = readYaml('src/tests/dm_1')
        dm = BoxDM()
        dm.create(config['domain']['box-mesh'])
        dm.setFemIndexing(config['domain']['ngl'])
        self.dm = dm

    def test_set_up_onlyFS(self):
        valsFS = {"velocity": [1,0], "vorticity": [0]}
        testData = {"free-slip": {
                "down": valsFS,
                "right": valsFS,
                "left": valsFS,
                "up": valsFS}}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)
        assert "FS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

    def test_set_up_custom_func(self):
        custFS = {"name": 'taylor_green', "attributes": ['velocity', 'vorticity']}
        testData = {"custom-func": custFS}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)
        assert "FS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

        bordersThatNeedsCoords = bcs.getBordersNeedsCoords()
        for bName in self.bcNames:
            assert bName in bordersThatNeedsCoords

    def test_set_up_custom_and_uniform(self):
        valsFS = {"velocity": [1,0], "vorticity": [0]}
        custFS = {"name": 'taylor_green', "attributes": ['velocity', 'vorticity']}

        testData = {"free-slip": {
                "down": valsFS,
                "right": {"custom-func": custFS},
                "left": {"custom-func": custFS},
                "up": valsFS}}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)
        assert "FS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

        bordersThatNeedsCoords = bcs.getBordersNeedsCoords()
        assert "right" in bordersThatNeedsCoords
        assert "left" in bordersThatNeedsCoords

    def test_set_up_onlyNS(self):
        valsNS = {"velocity": [1,0]}

        testData = {"no-slip": {
                "down": valsNS,
                "right": valsNS,
                "left": valsNS,
                "up": valsNS}}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)
        assert "NS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsNSNames) == len(self.bcNames)
        assert bcsFSNames == []

    def test_set_up_FSNS(self):

        valsFS = {"velocity": [1,0], "vorticity": [0]}
        custFS = {"custom-func":{"name": 'taylor_green', "attributes": ['velocity', 'vorticity']}}
        valsNS = {"velocity": [1,0]}

        testData = {"free-slip": {
                "down": valsFS,
                "right":custFS},
                    "no-slip": {
                "left": valsNS,
                "up": valsNS }}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)
        assert "FS-NS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
       
        assert 'down' in bcsFSNames
        assert 'right' in bcsFSNames
        assert 'up' in bcsNSNames
        assert 'left' in bcsNSNames

        assert 'right' in bcs.getBordersNeedsCoords()

    def test_get_indices(self):
        nodes_down = [0 , 1 , 2 , 3]
        nodes_right = [3 , 4, 5, 6]
        nodes_up = [6 , 7 , 8]
        nodes_left = [8 , 9 , 10 , 11, 0]

        valsFS = {"velocity": [1,0], "vorticity": [0]}
        valsNS = {"velocity": [2,0]}

        testData = {"free-slip": {
        "down": valsFS,
        "right": valsFS},
            "no-slip": {
        "left": valsNS,
        "up": valsNS}}

        bcs = BoundaryConditions(self.dm)
        bcs.setBoundaryConditions(testData)

        bcs.setBoundaryNodes("down", nodes_down)
        bcs.setBoundaryNodes("up", nodes_up)
        bcs.setBoundaryNodes("left", nodes_left)
        bcs.setBoundaryNodes("right", nodes_right)

        # no slip are left and up so...
        ns_nodes = nodes_left + nodes_up
        dim = 2
        ns_indices = [n*dim + dof for n in ns_nodes for dof in range(dim)]
        fs_nodes = nodes_down + nodes_right
        fs_indices = [n*dim + dof for n in fs_nodes for dof in range(dim)]

        assert set(ns_indices) == bcs.getNoSlipIndices()
        assert set(fs_indices) == bcs.getFreeSlipIndices()

        assert set(ns_indices) == bcs.getIndicesByType('no-slip')
        assert set(fs_indices) == bcs.getIndicesByType('free-slip')
