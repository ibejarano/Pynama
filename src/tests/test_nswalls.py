from common.nswalls import NoSlipWalls
import numpy as np
import numpy.testing as np_test
import unittest

class Test2DWalls(unittest.TestCase):
    def test_walls_with_velocity_x(self):
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns_1 = NoSlipWalls(lower=lower, upper=upper)
        ns_1.setWallVelocity(name="up", vel=[1,0])
        ns_1.setWallVelocity(name="down", vel=[2,0])
        self.assertIn("up", ns_1.getWallsWithVelocity())
        self.assertIn("down", ns_1.getWallsWithVelocity())

        for sideStatic in ["left", "right"]:
            self.assertNotIn(sideStatic, ns_1.getWallsWithVelocity())

    def test_walls_with_velocity_y(self):
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns_1 = NoSlipWalls(lower=lower, upper=upper)
        ns_1.setWallVelocity(name="left", vel=[0,1])
        ns_1.setWallVelocity(name="right", vel=[0,4])
        self.assertIn("left", ns_1.getWallsWithVelocity())
        self.assertIn("right", ns_1.getWallsWithVelocity())

        for sideStatic in ["up", "down"]:
            self.assertNotIn(sideStatic, ns_1.getWallsWithVelocity())

    def test_ignore_velocity_normal(self):
        """this tests tries to assert that if i set a vel that is normal it ignores it"""
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns_ud = NoSlipWalls(lower=lower, upper=upper)
        ns_lr = NoSlipWalls(lower=lower, upper=upper)

        ns_lr.setWallVelocity(name="left", vel=[1,0])
        ns_lr.setWallVelocity(name="right", vel=[1,0])
        ns_ud.setWallVelocity(name="up", vel=[0,1])
        ns_ud.setWallVelocity(name="down", vel=[0,1])

        for sideStatic in ["up", "down"]:
            self.assertNotIn(sideStatic, ns_ud.getWallsWithVelocity())
        
        for sideStatic in ["left", "right"]:
            self.assertNotIn(sideStatic, ns_lr.getWallsWithVelocity())

    def test_velocities_x_and_dof(self):
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns_lr = NoSlipWalls(lower=lower, upper=upper)

        vel_test = 4
        dof_setting = 1
        ns_lr.setWallVelocity(name="right", vel=[0,vel_test])
        vel, velDofs = ns_lr.getWallVelocity("right")
        self.assertEqual(vel_test, vel[0])
        self.assertEqual(dof_setting, velDofs[0])

        vel_test = 8
        ns_lr.setWallVelocity(name="left", vel=[0,vel_test])
        vel, velDofs = ns_lr.getWallVelocity("left")
        self.assertEqual(vel_test, vel[0])
        self.assertEqual(dof_setting, velDofs[0])

    def test_velocities_y_and_dof(self):
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns_ud = NoSlipWalls(lower=lower, upper=upper)

        vel_test = 3
        dof_setting = 0
        ns_ud.setWallVelocity(name="up", vel=[vel_test, 0])
        vel, velDofs = ns_ud.getWallVelocity("up")
        self.assertEqual(vel_test, vel[0])
        self.assertEqual(dof_setting, velDofs[0])

        vel_test = 9
        ns_ud.setWallVelocity(name="down", vel=[vel_test, 0])
        vel, velDofs = ns_ud.getWallVelocity("down")
        self.assertEqual(vel_test, vel[0])
        self.assertEqual(dof_setting, velDofs[0])

    def test_normal_computing(self):
        lower= np.random.rand(2)
        upper = np.random.rand(2) + lower
        ns = NoSlipWalls(lower=lower, upper=upper)
        
        for wall in ["left", "right"]:
            normal = 0  # x-direction = 0
            self.assertEqual(normal, ns.getWalletNormalBySideName(wall))

        for wall in ["up", "down"]:
            normal = 1 # y-direction = 0
            self.assertEqual(normal, ns.getWalletNormalBySideName(wall))

    def test_naming_convention(self):
        pass


class Test3DWalls(unittest.TestCase):
    def setUp(self):
        pass


