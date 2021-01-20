import unittest
from cases.base_problem import BaseProblemTest as FemProblem
import yaml
from petsc4py import PETSc
import numpy as np

class TestKleUniform2D(unittest.TestCase):
    caseYaml = 'uniform'
    caseOpts = {'lower':[0,0],'upper':[1,1],'nelem':[3,3], 'keepCoords': True}
    def setUp(self):
        with open(f'src/cases/{self.caseYaml}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        fem = FemProblem(yamlData, case=self.caseYaml, **self.caseOpts)
        fem.setUp()
        fem.setUpSolver()
        self.fem = fem

    def test_solveKLE(self):
        exactVel, exactVort = self.fem.generateExactVecs(vel=[4,0], vort=[0])
        self.fem.solverKLE.solve(vort=exactVort)

        vel = self.fem.solverKLE.getSolution()
        error = exactVel - vel
        normError = error.norm(norm_type=2)
        self.assertLess(normError, 1e-12)

class TestKleFunc2D(TestKleUniform2D):
    caseYaml = 'taylor-green'
    caseOpts = {'lower':[0,0],'upper':[1,1],'nelem':[10,10], "ngl": 5, 'keepCoords': True}
    def test_solveKLE(self):
        exactVel, exactVort = self.fem.generateExactVecs(time=0.0)
        self.fem.solverKLE.solve(vort=exactVort)

        vel = self.fem.solverKLE.getSolution()
        error = exactVel - vel
        normError = error.norm(norm_type=2)
        self.assertLess(normError, 1e-5)

# class TestKleUniform3D(TestKleUniform2D):
#     case = 'uniform'
#     cte = [1,0,0]
#     caseOpts = {'lower':[0,0,0],'upper':[1,1,1],'nelem':[2,2,2], "freeSlip": { 'down': cte, 'up': cte, 'back': cte, 'front': cte, 'left': cte, 'right': cte } }


class TestRHSEval(unittest.TestCase):
    caseYaml = 'uniform'
    caseOpts = {'lower':[0,0],'upper':[1,1],'nelem':[2,2], 'ngl': 2, 'keepCoords': True}
    
    def setUp(self):
        with open(f'src/cases/{self.caseYaml}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        fem = FemProblem(yamlData, case=self.caseYaml, **self.caseOpts)
        fem.setUp()
        fem.setUpSolver()
        self.fem = fem

    def test_VtensV_eval(self):
        domain = {'lower':[0,0],'upper':[1,1],'nelem':[2,2], 'ngl':2}

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

        self.fem.computeVtensV(vec=vec_init)

        np.testing.assert_array_almost_equal(vec_ref, self.fem._VtensV, decimal=10)