from domain.boundaries.boundary import FunctionBoundary
from functions.taylor_green import velocity_test, vorticity_test

import unittest
import numpy as np
import numpy.testing as np_test

class TestBoundaryFunctionTaylorGreen2D(unittest.TestCase):
    custom_func = 'taylor_green'
    attrs =  ['velocity', 'vorticity', 'alpha']
    nnodes = 250
    dim = 2
    def setUp(self):
        x = np.linspace(0,1,self.nnodes)
        y = np.repeat(1.0, self.nnodes)
        self.coords = np.dstack((x,y))[0]
        self.nodes = np.arange(self.nnodes, dtype=np.int32)
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
        nu = 1
        vels = self.b.getValues("velocity", t , nu)

        vels_ref = np.zeros(len(self.nodes)*self.dim)
        for node, coord in enumerate(self.coords):
            val = velocity_test(coord, t, nu)
            vels_ref[ node*self.dim : node*self.dim + self.dim ] =  val

        np_test.assert_almost_equal(vels, vels_ref, decimal=14)

    def test_get_nodes_vorticities(self):
        assert self.dim == 2
        t = 0
        nu = 1
        vort = self.b.getValues("vorticity", t , nu)

        vort_ref = np.zeros(len(self.nodes))
        for node, coord in enumerate(self.coords):
            val = vorticity_test(coord, t, nu)
            vort_ref[ node: node+1 ] =  val

        np_test.assert_almost_equal(vort, vort_ref, decimal=14)

class TestBoundaryFunctionTaylorGreen2DCaseA(TestBoundaryFunctionTaylorGreen2D):
    custom_func = 'taylor_green'
    attrs =  ['velocity', 'vorticity', 'alpha']
    nnodes = 33
    dim = 2

class TestBoundaryFunctionTaylorGreen2DCaseB(TestBoundaryFunctionTaylorGreen2D):
    custom_func = 'taylor_green'
    attrs =  ['velocity', 'vorticity', 'alpha']
    nnodes = 500
    dim = 2