import unittest
from cases.uniform import UniformFlow
from cases.custom_func import CustomFuncCase
import yaml
from petsc4py import PETSc
import numpy as np

class TestKle2D(unittest.TestCase):
    def setFemProblem(self, case, **kwargs):
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        if case == 'uniform':
            fem = UniformFlow(yamlData, case=case, **kwargs)
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

class TestRHSEval(TestKle2D):

    def test_VtensV_eval(self):
        domain = {'lower':[0,0,0],'upper':[1,1],'nelem':[2,2], 'ngl':2}
        fem = self.setFemProblem('uniform', **domain)

        vec_init = [ 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ]
        vec_ref = [ 1 , 1*2 , 2*2 ,
        3*3 , 3*4 , 4*4 ,
        5*5 , 5*6 , 6*6 ,
        7*7 , 7*8 , 8*8 ,
        9*9 , 9*10 ,10*10  ,
        11*11 ,11*12  , 12*12 ,
        13*13 , 13*14 , 14*14 ,
        15*15 , 15*16 , 16*16 ,
        17*17 , 17*18 , 18*18 ,
        ]

        vec_ref = np.array(vec_ref)
        vec_init = PETSc.Vec().createWithArray(np.array(vec_init))

        fem.computeVtensV(vec=vec_init)

        np.testing.assert_array_almost_equal(vec_ref, fem._VtensV, decimal=10)