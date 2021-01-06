from boundaries.boundary_conditions import BoundaryConditions
from boundaries.boundary import Boundary
import unittest
import numpy as np
import numpy.testing as np_test

class TestBasicBoundary(unittest.TestCase):
    vel = [1, 4]
    vort = [ 0 ]
    dim = 2
    nodesInBorder =  [0, 11, 24, 78]

    def setUp(self):
        boundary = Boundary("test", "free-slip", self.dim)
        boundary.setValues('velocity', self.vel)
        boundary.setValues("vorticity", self.vort)
        self.boundary = boundary

    def test_create_basic_boundary(self):
        # boundary = Boundary("test", "free-slip", [1, 4])
        assert "test" == self.boundary.getName()
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

class TestBasicBoundary3D(TestBasicBoundary):
    vel = [1, 6 , 8 ]
    vort = [ 32, 12, 124 ]
    dim = 3
    nodesInBorder =  [0, 11, 24, 78]

class TestBoundaryConditions(unittest.TestCase):
    def setUp(self):
        self.bcNames = ['up','down', 'right', 'left']

    def test_set_up_onlyFS(self):
        testData = {"free-slip": {
                "down": [None, 0],
                "right": [1, 0],
                "left": [1, 0],
                "up": [1, 1]}}
        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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

class TestBoundaryFunctionTaylorGreen2D(unittest.TestCase):
    custom_func = 'taylor_green'
    attrs =  ['velocity', 'vorticity', 'alpha']
    coords = np.array([0,0, 0.1, 0.1 , 0.3, 0.3 , 0.6, 0.6, 0.8, 0.8])
    nodes = [0,1,2,3,4]
    dim = 2
    def setUp(self):
        b = FunctionBoundary('up', self.custom_func, self.attrs ,dim=self.dim )
        self.b = b
        self.b.setNodes(self.nodes)
        self.b.setNodesCoordinates(self.coords)

    def test_get_coords(self):
        coords = self.b.getNodesCoordinates()
        np_test.assert_almost_equal(coords, self.coords, decimal=14)

    def test_get_nodes_velocities(self):
        vels = self.b.getValues("velocity", 0 , 0.1)
        raise Exception("Need Test case")

    def test_get_nodes_vorticities(self):
        vorts = self.b.getValues("vorticity", 0 , 0.1)
        raise Exception("Need Test case")

class TestBoundaryFunctionTaylorGreen3D(unittest.TestCase):
    custom_func = 'taylor_green_3d'
    sideNames = ['up', 'right', 'left', 'down', 'front', 'back']
    coords = np.array([0,0,0, 0.5, 0.5,0.5, 2,0,0])
    dim = 3