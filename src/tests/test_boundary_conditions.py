from boundaries.boundary_conditions import BoundaryConditions
from boundaries.boundary import Boundary, FunctionBoundary
from functions.taylor_green import velocity_test, vorticity_test

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




class TestBasicBoundary3D(TestBasicBoundary):
    vel = [1, 6 , 8 ]
    vort = [ 32, 12, 124 ]
    dim = 3
    nodesInBorder =  [0, 11, 24, 78]

class TestBoundaryConditions(unittest.TestCase):
    bcNames = ['up','down', 'right', 'left']

    def test_set_up_onlyFS(self):
        valsFS = {"velocity": [1,0], "vorticity": [0]}
        testData = {"free-slip": {
                "down": valsFS,
                "right": valsFS,
                "left": valsFS,
                "up": valsFS}}

        bcs = BoundaryConditions(self.bcNames)
        bcs.setBoundaryConditions(testData)
        assert "FS" == bcs.getType()

        bcsFSNames = bcs.getNamesByType('free-slip')
        bcsNSNames = bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

    def test_set_up_custom_func(self):
        custFS = {"name": 'taylor_green', "attributes": ['velocity', 'vorticity']}
        testData = {"custom-func": custFS}

        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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

        bcs = BoundaryConditions(self.bcNames)
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
        
        self.coords.reshape((len(self.nodes), self.dim))

    def test_get_coords(self):
        coords = self.b.getNodesCoordinates()
        np_test.assert_almost_equal(coords, self.coords, decimal=14)

    def test_get_nodes_velocities(self):
        t = 0
        nu = 100
        vels = self.b.getValues("velocity", t , nu)


        vels_ref = np.zeros(len(self.nodes)*self.dim)
        for node, coord in enumerate(self.coords):
            val = velocity_test(coord, t, nu)
            vels_ref[ node*self.dim : node*self.dim + self.dim ] =  val

        np_test.assert_almost_equal(vels, vels_ref, decimal=14)

    def test_get_nodes_vorticities(self):
        assert self.dim == 2
        t = 0
        nu = 100
        vels = self.b.getValues("vorticity", t , nu)

        vels_ref = np.zeros(len(self.nodes))
        for node, coord in enumerate(self.coords):
            val = vorticity_test(coord, t, nu)
            vels_ref[ node: node+1 ] =  val

        np_test.assert_almost_equal(vels, vels_ref, decimal=14)

class TestBoundaryFunctionTaylorGreen3D(unittest.TestCase):
    custom_func = 'taylor_green_3d'
    sideNames = ['up', 'right', 'left', 'down', 'front', 'back']
    coords = np.array([0,0,0, 0.5, 0.5,0.5, 2,0,0])
    dim = 3