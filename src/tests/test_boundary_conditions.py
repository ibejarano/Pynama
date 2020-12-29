from domain.boundary_conditions import Boundary, BoundaryConditions
import unittest
import numpy as np
import numpy.testing as np_test


class TestBoundary(unittest.TestCase):
    def test_create_basic_boundary(self):
        boundary = Boundary("test", "free-slip", [1, 4])
        assert "test" == boundary.getName()
        assert "free-slip" == boundary.getType()
        np_test.assert_equal(boundary.getValues(), np.array([1, 4]))

    def test_create_basic3d_boundary(self):
        boundary = Boundary("test", "free-slip", [1, 2, 3])
        assert "test" == boundary.getName()
        assert "free-slip" == boundary.getType()
        np_test.assert_equal(boundary.getValues(), np.array([1, 2, 3]))

    def test_dofs_constrained_2d(self):
        desired_suite = [([1, None], np.array([1])),
                         ([None, 4], np.array([4]))]
        for d in desired_suite:
            vals, desired_val = d
            boundary = Boundary("test", "free-slip", vals)
            np_test.assert_equal(boundary.getValues(), desired_val)

    def test_set_get_dofs_constrained(self):
        nodesBC = [0, 11, 24, 78]
        valuesBC = [1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)

        # This case has both dofs constrained
        dim = len(valuesBC)
        dofsBC_desired = [i*2 + dof for i in nodesBC for dof in range(dim)]
        np_test.assert_equal(boundary.getDofsConstrained(), dofsBC_desired)

    def test_get_nodes(self):
        nodesBC = [123, 12415, 1566, 121]
        valuesBC = [1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)
        np_test.assert_equal(boundary.getNodes(), nodesBC)

    def test_destroy(self):
        nodesBC = [0, 11, 24, 78]
        valuesBC = [1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)
        IS = boundary.getIS()
        boundary.destroy()
        assert IS.getRefCount() == 0


class TestBoundaryConditions(unittest.TestCase):
    def test_set_up_onlyFS(self):
        testData = {"free-slip": {
                "down": [None, 0],
                "right": [1, 0],
                "left": [1, 0],
                "up": [1, 1]}}

        bcs = BoundaryConditions()
        bcs.setBoundaryConditions(testData)
        assert "only FS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == 4
        assert bcsNSNames == []

    def test_set_up_onlyNS(self):
        testData = {"no-slip": {
                "down": [None, 0],
                "right": [1, 0],
                "left": [1, 0],
                "up": [1, 1]}}

        bcs = BoundaryConditions()
        bcs.setBoundaryConditions(testData)
        assert "only NS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsNSNames) == 4
        assert bcsFSNames == []

    def test_set_up_FSNS(self):
        testData = {"free-slip": {
                "down": [None, 0],
                "right": [1, 0]},
                    "no-slip": {
                "left": [1, 0],
                "up": [1, 1]}}

        bcs = BoundaryConditions()
        bcs.setBoundaryConditions(testData)
        assert "FS NS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
       
        assert 'down' in bcsFSNames
        assert 'right' in bcsFSNames
        assert 'up' in bcsNSNames
        assert 'left' in bcsNSNames

    def test_get_indices(self):
        nodes_down = [0 , 1 , 2 , 3]
        nodes_right = [3 , 4, 5, 6]
        nodes_up = [6 , 7 , 8]
        nodes_left = [8 , 9 , 10 , 11, 0]

        testData = {"free-slip": {
        "down": [None, 0],
        "right": [1, 0]},
            "no-slip": {
        "left": [1, 0],
        "up": [1, 1]}}

        bcs = BoundaryConditions()
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
