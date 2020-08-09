import unittest
import numpy as np
from domain.immersed_body import BodiesContainer
from math import ceil, pi

class TestBodyGeneration(unittest.TestCase):
    def test_single_circle(self):
        circle = BodiesContainer('single')
        dh = 0.01
        circle.createBodies(dh)
        longCircle = 2*pi*0.5
        divs = ceil(longCircle / dh)
        dh = longCircle / divs
        dh_test = circle.getElementLong()
        nodes = circle.getTotalNodes()
        center = circle.getCenters()
        assert nodes == divs 
        assert dh == dh_test
        for i in range(1):
            assert center[i] == [ 0 , 0 ]

    def test_side_by_side(self):
        sideBySide = BodiesContainer('side-by-side')
        dh = 0.1
        sideBySide.createBodies(dh)
        longCircle = 2*pi*0.5
        divs = ceil(longCircle / dh) 
        dh = longCircle / divs
        divs *= 2
        assert divs == sideBySide.getTotalNodes()
        assert dh == sideBySide.getElementLong()

        test_centers = sideBySide.getCenters()
        assert test_centers[0] == [ 0 , -1 ]
        assert test_centers[1] == [ 0 , 1 ]
        for i in range(int(divs/2)):
            coords_A = sideBySide.getNodeCoordinates(i)
            coords_B = sideBySide.getNodeCoordinates(i+int(divs/2))
            assert coords_A[0] == coords_B[0]
            np.testing.assert_almost_equal(coords_A[1] +1, coords_B[1] -1, 11)

    def test_tandem(self):
        tandem = BodiesContainer('tandem')
        dh = 0.01
        tandem.createBodies(dh)
        longCircle = 2*pi*0.5
        divs = ceil(longCircle / dh) 
        dh = longCircle / divs
        divs *= 2
        assert divs == tandem.getTotalNodes()
        assert dh == tandem.getElementLong()

        test_centers = tandem.getCenters()
        assert test_centers[0] == [ -1 , 0 ]
        assert test_centers[1] == [ 1 , 0 ]
        for i in range(int(divs/2)):
            coords_A = tandem.getNodeCoordinates(i)
            coords_B = tandem.getNodeCoordinates(i+int(divs/2))
            np.testing.assert_almost_equal(coords_A[0] + 1, coords_B[0] - 1, 11)
            np.testing.assert_almost_equal(coords_A[1], coords_B[1], 11)