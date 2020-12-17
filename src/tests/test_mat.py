from cases.uniform import UniformFlow
import unittest
import yaml
import numpy as np
import numpy.testing as np_test
from matrices.mat_generator import Mat

class TestMat2D(unittest.TestCase):
    def setFemProblem(self, case, **kwargs):
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        if case == 'uniform':
            fem = UniformFlow(yamlData, case=case, **kwargs)
        else:
            # fem = CustomFuncCase(yamlData, case=case , **kwargs)
            raise Exception("Test case not implemented")
        fem.setUpDomain()
        # fem.setUpBoundaryConditions()
        # fem.setUpEmptyMats()
        return fem

    def test_create_nnz(self):
        domain = {'lower':[0,0],'upper':[1,1],'nelem':[2,2], 'ngl':3}
        fem = self.setFemProblem('uniform', **domain)
        mat = Mat(2)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = fem.dom.getMatIndices()
        for i in range(1,3):
            for j in range(1,3):
                d_ref, o_ref = mat.createNonZeroIndex(d_nnz_ind, o_nnz_ind, i , j)
                d_test, o_test = mat.createNNZWithArray(d_nnz_ind, o_nnz_ind, i , j)
                np_test.assert_array_equal(d_ref, d_test)
                np_test.assert_array_equal(o_ref, o_test)