import unittest
from cases.custom_func import CustomFuncCase
import numpy as np
import numpy.testing as np_test
import h5py
from petsc4py import PETSc

class TaylorGreenTest(unittest.TestCase):
    def setUp(self):
        self.fem = CustomFuncCase()
        self.fem.setUp()
        self.fem.setUpSolver()

        _, self.vort_test =  self.fem.generateExactVecs(time=0.01)
 
    # def test_curl_operation(self):
    #     # nabla x v = w
    #     # curl . v = w
    #     vel = self.fem.mat.K.createVecRight()
    #     allNodes = self.fem.dom.getAllNodes()
    #     vel = self.fem.dom.applyValuesToVec(allNodes, [1,2], vel)
    #     vort = self.fem.mat.Rw.createVecRight()
    #     self.fem.mat.Curl.mult(vel, vort)
    #     # Testing # 1: Uniform flow
    #     np_test.assert_array_almost_equal(np.zeros(vort.getSize()), vort.getArray(), decimal=15)

    # def test_rhs_operation(self):
    #     mu = 1
    #     rho = 1
    #     vel = self.fem.mat.K.createVecRight()
    #     allNodes = self.fem.dom.getAllNodes()
    #     vel = self.fem.dom.applyValuesToVec(allNodes, [1,2], vel)
    #     sK, eK = self.fem.mat.K.getOwnershipRange()
    #     for indN in range(sK, eK, self.fem.dim):
    #         indicesVV = [indN * self.fem.dim_s / self.fem.dim + d
    #                      for d in range(self.fem.dim_s)]
    #         VelN = vel.getValues([indN + d for d in range(self.fem.dim)])
    #         VValues = [VelN[0] ** 2, VelN[0] * VelN[1], VelN[1] ** 2]

    #         self.fem._VtensV.setValues(indicesVV, VValues, False)

    #     self.fem._VtensV.assemble()

    #     # self._Aux1 = self.SrT * self._Vel
    #     self.fem.mat.SrT.mult(vel, self.fem._Aux1)

    #     # _Aux1 = 2*Mu * S - rho * Vvec ^ VVec
    #     self.fem._Aux1 *= (2.0 * mu)
    #     self.fem._Aux1.axpy(-1.0 * rho, self.fem._VtensV)
    #     self.fem._Aux1.view()

    #     # FIXME: rhs should be created previously or not?
    #     rhs = vel.duplicate()
    #     # RHS = Curl * Div(SrT) * 2*Mu * S - rho * Vvec ^ VVec
    #         # rhs = (self.DivSrT * self._Aux1) / self.rho
    #     self.fem.mat.DivSrT.mult(self.fem._Aux1, rhs)
    #     rhs.scale(1/rho)
        
    #     dwdt = self.fem.mat.Rw.createVecRight()
    #     self.fem.mat.Curl.mult(rhs, dwdt)
    #     np_test.assert_array_almost_equal(np.zeros(dwdt.getSize()), dwdt.getArray(), decimal=15)

    def test_vort(self):
        loadh5 = h5py.File('src/tests/operators-vecs/vort.h5','r')
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        vort_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['vort'][:])

        np_test.assert_array_almost_equal(vort_ref.max(), self.vort_test.max(), decimal=15)

    def test_vel(self):
        loadh5 = h5py.File('src/tests/operators-vecs/vel.h5','r')
        vel_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['vel'][:])
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        loadh5 = h5py.File('src/tests/operators-vecs/vort.h5','r')
        vort_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['vort'][:])
        self.fem.solveKLE(0.01, vort_ref)
        np_test.assert_array_almost_equal(vel_ref.max(), self.fem.vel.max(), decimal=15)

    def test_vtensv(self):
        loadh5 = h5py.File('src/tests/operators-vecs/vtensv.h5','r')
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        vtensv_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['VtensV'][:])

        # vtensv_ref.view()
        # vel = self.fem.mat.K.createVecRight()
        # allNodes = self.fem.dom.getAllNodes()
        # vel = self.fem.dom.applyValuesToVec(allNodes, [1,2], vel)
        
    def test_srt_operator(self):
        loadh5 = h5py.File('src/tests/operators-vecs/srt.h5','r')
        srt_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['SrT'][:])
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        loadh5 = h5py.File('src/tests/operators-vecs/vel.h5','r')
        vel_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['vel'][:])
        srt_test = srt_ref.duplicate()

        self.fem.mat.SrT.mult(vel_ref, srt_test)
        
        np_test.assert_array_almost_equal(srt_ref.max(), srt_test.max(), decimal=15)

    def test_divsrt_operator(self):
        loadh5 = h5py.File('src/tests/operators-vecs/divsrt.h5','r')
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        divsrt_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['DivSrT'][:])
        divsrt_ref.view()

    def test_all_operator(self):
        loadh5 = h5py.File('src/tests/operators-vecs/rhs.h5','r')
        # vtensv_hdf5 = np.array(vtensvLoad['fields']['VtensV'][:])
        rhs_ref = PETSc.Vec().createWithArray(array=loadh5['fields']['rhs'][:])
        rhs_ref.view()