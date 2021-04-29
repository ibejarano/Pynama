from domain.boundaries.boundary import Boundary

import unittest
import numpy as np
import numpy.testing as np_test

class TestNoSlipBoundary(unittest.TestCase):
    vel = [1, 4]
    dim = 2
    nodesInBorder =  [0, 11, 24, 78]

    def setUp(self):
        boundaryDown = Boundary("down", "no-slip", self.dim)
        boundaryDown.setValues('velocity', self.vel)
        self.boundaryDown = boundaryDown
        self.boundaryDown.setNodes(self.nodesInBorder)


        boundaryLeft = Boundary("left", "no-slip", self.dim)
        boundaryLeft.setValues('velocity', self.vel)
        self.boundaryLeft = boundaryLeft
        self.boundaryLeft.setNodes(self.nodesInBorder)

    def test_ns_normals(self):
        normalsDofs_test_down = self.boundaryDown.getNormalDofs()
        normalsDofs_test_left = self.boundaryLeft.getNormalDofs()

        # the side is down so normal is y or 1
        normalDof_down = 1
        ref_down = [ node*self.dim + normalDof_down for node in self.nodesInBorder ]
        # left normal is x or 0
        normalDof_left = 0
        ref_left = [ node*self.dim + normalDof_left for node in self.nodesInBorder ]

        assert normalsDofs_test_down == set(ref_down)
        assert normalsDofs_test_left == set(ref_left)

    def test_ns_tangentials(self):
        tanDofs_test_down = self.boundaryDown.getTangDofs()
        tanDofs_test_left = self.boundaryLeft.getTangDofs()
        
        tanDof = 0
        ref_down = [ node*self.dim + tanDof for node in self.nodesInBorder ]

        tanDof = 1
        ref_left = [ node*self.dim + tanDof for node in self.nodesInBorder ]

        assert tanDofs_test_down == set(ref_down)
        assert tanDofs_test_left == set(ref_left)