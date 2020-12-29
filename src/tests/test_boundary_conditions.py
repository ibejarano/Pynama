from domain.boundary_conditions import Boundary , BoundaryConditions
import unittest
import numpy as np
import numpy.testing as np_test

class TestBoundary(unittest.TestCase):
    def test_create_basic_boundary(self):
        boundary = Boundary("test", "free-slip", [1, 4])
        assert "test" == boundary.getName()
        assert "free-slip" == boundary.getType()
        np_test.assert_equal(boundary.getValues(), np.array([1,4]))

    def test_create_basic3d_boundary(self):
        boundary = Boundary("test", "free-slip", [1, 2, 3])
        assert "test" == boundary.getName()
        assert "free-slip" == boundary.getType()
        np_test.assert_equal(boundary.getValues(), np.array([1,2, 3]))

    def test_dofs_constrained_2d(self):
        desired_suite = [  ([1, None] , np.array([1])) , ([None, 4] , np.array([4])) ]   
        for d in desired_suite:
            vals, desired_val = d
            boundary = Boundary("test", "free-slip", vals)
            np_test.assert_equal(boundary.getValues(), desired_val)

    def test_set_get_dofs_constrained(self):
        nodesBC = [0 , 11, 24, 78 ]
        valuesBC = [ 1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)

        # This case has both dofs constrained
        dim = len(valuesBC)
        dofsBC_desired = [ i*2 + dof for i in nodesBC for dof in range(dim)  ]
        np_test.assert_equal(boundary.getDofsConstrained(), dofsBC_desired)

    def test_get_nodes(self):
        nodesBC = [ 123 , 12415, 1566, 121 ]
        valuesBC = [ 1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)
        np_test.assert_equal(boundary.getNodes(), nodesBC)

    def test_destroy(self):
        nodesBC = [0 , 11, 24, 78 ]
        valuesBC = [ 1, 4]
        boundary = Boundary("test", "free-slip", valuesBC)
        boundary.setNodes(nodesBC)
        IS = boundary.getIS()
        boundary.destroy()
        assert IS.getRefCount() == 0
