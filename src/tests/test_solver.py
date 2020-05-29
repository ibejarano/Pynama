import unittest
from cases.uniform import UniformFlow

class UniformFlowTest(unittest.TestCase):
    def setUp(self):
        self.fem = UniformFlow()
        self.fem.setUpSolver()
        
    def test_solveKLE(self):
        exactVel = self.fem.generateExactVel()
        self.fem.solve()
        error = exactVel - self.fem.vel
        normError = error.norm(norm_type=2)
        self.assertLess(normError, 1e-14)