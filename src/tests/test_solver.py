import unittest
from cases.uniform import UniformFlow
from cases.custom_func import CustomFuncCase
import yaml

class TestKle2D(unittest.TestCase):
    def setFemProblem(self, case, **kwargs):
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        if case == 'uniform':
            fem = UniformFlow(yamlData, case=case)
        else:
            fem = CustomFuncCase(yamlData, case=case , **kwargs)
        fem.setUp()
        fem.setUpSolver()
        return fem

    def test_solveKLE_uniform(self):
        fem = self.setFemProblem('uniform')
        exactVel, exactVort = fem.generateExactVecs()
        fem.solveKLE(time=0.0, vort=exactVort)
        error = exactVel - fem.vel
        normError = error.norm(norm_type=2)
        self.assertLess(normError, 1e-12)
        del fem

    def test_solveKLE_taylorgreen(self):
        domain = {'nelem':[2,2], 'ngl':11}
        fem = self.setFemProblem('taylor-green', **domain)
        exactVel, exactVort = fem.generateExactVecs(0.0)
        fem.solveKLE(time=0.0, vort=exactVort)
        error = exactVel - fem.vel
        normError = error.norm(norm_type=2)
        self.assertLess(normError, 2e-8)
        del fem

class TestKle3D(unittest.TestCase):
    def setFemProblem(self, case, **kwargs):
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        if case == 'uniform':
            fem = UniformFlow(yamlData, case=case, **kwargs)
        else:
            # fem = CustomFuncCase(yamlData, case=case , **kwargs)
            raise Exception("Test case not implemented")
        fem.setUp()
        fem.setUpSolver()
        return fem

    def test_solveKLE_uniform(self):
        domain = {'lower':[0,0,0],'upper':[1,1,1],'nelem':[3,3,3], 'ngl':3}
        fem = self.setFemProblem('uniform', **domain)
        exactVel, exactVort = fem.generateExactVecs()
        fem.solveKLE(time=0.0, vort=exactVort)
        error = exactVel - fem.vel
        normError = error.norm(norm_type=2)
        # error.view()
        fem.view()
        self.assertLess(normError, 2e-13)
        del fem