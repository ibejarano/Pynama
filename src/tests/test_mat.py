from cases.uniform import UniformFlow
from domain.domain import Domain
import unittest
import yaml
import numpy as np
import numpy.testing as np_test
from matrices.mat_generator import Mat

class TestMat2D(unittest.TestCase):

    def setUp(self):
        box = {'lower':[0,0],'upper':[1,1],'nelem':[2,2]}
        domain = {"domain": {"box-mesh": box, "ngl": 3}}
        dom = Domain()
        dom.configure(domain)
        dom.create()
        dom.setUpIndexing()
        dim = 2
        self.mat = Mat(dim)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = dom.getMatIndices()
        self.rStart = rStart
        self.rEnd = rEnd
        self.d_nnz_ind = d_nnz_ind
        self.o_nnz_ind = o_nnz_ind
        self.ind_d = ind_d
        self.ind_o = ind_o

    def setFemProblem(self, case, **kwargs):
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
        if case == 'uniform':
            fem = UniformFlow(yamlData, case=case, **kwargs)
        else:
            raise Exception("Test case not implemented")
        fem.setUpDomain()
        return fem

    def test_create_nnz(self):
        for i in range(1,3):
            for j in range(1,3):
                d_ref, o_ref = self.mat.createNonZeroIndex(self.d_nnz_ind, self.o_nnz_ind, i , j)
                d_test, o_test = self.mat.createNNZWithArray(self.d_nnz_ind, self.o_nnz_ind, i , j)
                np_test.assert_array_equal(d_ref, d_test)
                np_test.assert_array_equal(o_ref, o_test)

    def test_nnz_diag_off_quantities(self):
        nnodes = (self.rEnd - self.rStart)
        # d_nnz_ref = [9, 9, 9, 9, 9, 15, 9, 15, 25, 15, 9, 15, 9, 9, 9, 15, 15, 9, 9, 9, 9, 15, 15, 9, 9]
        o_nnz_ref = [0] * nnodes
        quantities_test = set(self.d_nnz_ind)
        quantities_ref = {9, 15, 25}

        assert quantities_ref == quantities_test
        assert len(o_nnz_ref) == 25

    def test_nnz_counts(self):
        nodes_quantities = [9 , 15 , 25]
        counts_ref = np.array([ 16 , 8 , 1 ])
        nnodes= self.rEnd - self.rStart
        assert counts_ref.sum() == nnodes
        for i, q in enumerate(nodes_quantities):
            counts = list(self.d_nnz_ind).count(q)
            assert counts == counts_ref[i]

class TestMat3D(TestMat2D):
    def setUp(self):
        box = {'lower':[0,0,0],'upper':[1,1,1],'nelem':[2,2,2]}
        domain = {"domain": {"box-mesh": box, "ngl": 3}}
        dom = Domain()
        dom.configure(domain)
        dom.create()
        dom.setUpIndexing()
        dim = 3
        self.mat = Mat(dim)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = dom.getMatIndices()
        self.rStart = rStart
        self.rEnd = rEnd
        self.d_nnz_ind = d_nnz_ind
        self.o_nnz_ind = o_nnz_ind
        self.ind_d = ind_d
        self.ind_o = ind_o

    def test_create_nnz(self):
        for i in range(1,3):
            for j in range(1,3):
                d_ref, o_ref = self.mat.createNonZeroIndex(self.d_nnz_ind, self.o_nnz_ind, i , j)
                d_test, o_test = self.mat.createNNZWithArray(self.d_nnz_ind, self.o_nnz_ind, i , j)
                np_test.assert_array_equal(d_ref, d_test)
                np_test.assert_array_equal(o_ref, o_test)

    def test_nnz_diag_off_quantities(self):
        nnodes = (self.rEnd - self.rStart)
        o_nnz_ref = [0] * nnodes
        quantities_test = set(self.d_nnz_ind)
        quantities_ref = {27, 45, 75 ,125}

        assert quantities_ref == quantities_test
        assert len(o_nnz_ref) == (5*5*5)

    def test_nnz_counts(self):
        nodes_quantities = [27, 45, 75 ,125]
        counts_ref = np.array([64 , 48 , 12 , 1 ])
        nnodes= self.rEnd - self.rStart
        assert counts_ref.sum() == nnodes
        for i, q in enumerate(nodes_quantities):
            counts = list(self.d_nnz_ind).count(q)
            assert counts == counts_ref[i]