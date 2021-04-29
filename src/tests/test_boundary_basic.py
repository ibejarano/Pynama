from domain.boundaries.boundary import Boundary

import unittest
import numpy as np
import numpy.testing as np_test

class TestBasicBoundary(unittest.TestCase):
    vel = [1, 4]
    vort = [ 0 ]
    dim = 2
    nodesInBorder =  [0, 11, 24, 78]

    def setUp(self):
        boundary = Boundary("left", "free-slip", self.dim)
        boundary.setValues('velocity', self.vel)
        boundary.setValues("vorticity", self.vort)
        self.boundary = boundary

    def test_create_basic_boundary(self):
        assert "left" == self.boundary.getName()
        assert "free-slip" == self.boundary.getType()

    def test_set_get_dofs_constrained(self):
        nodesBC = [0, 11, 24, 78]
        self.boundary.setNodes(nodesBC)

        dofsBC_desired = [i*self.dim + dof for i in nodesBC for dof in range(self.dim)]

        np_test.assert_equal(self.boundary.getDofsConstrained(), dofsBC_desired)

    def test_get_nodes(self):
        nodesBC = [123, 12415, 1566, 121]

        self.boundary.setNodes(nodesBC)
        np_test.assert_equal(self.boundary.getNodes(), nodesBC)

    def test_get_values(self):
        self.boundary.setNodes(self.nodesInBorder)

        total_nodes_in_bc = len(self.nodesInBorder)
        desired_vel = list()
        for i in range(total_nodes_in_bc):
            for val in self.vel:
                desired_vel.append(val)

        test_vel = self.boundary.getValues('velocity')
    
        np_test.assert_almost_equal(test_vel, desired_vel, decimal=14)

    def test_destroy(self):
        nodesBC = [0, 11, 24, 78]
        self.boundary.setNodes(nodesBC)
        IS = self.boundary.getIS()
        self.boundary.destroy()
        assert IS.getRefCount() == 0