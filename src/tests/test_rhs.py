import unittest
from matrices.new_mat import Matrices
from domain.dmplex_bc import NewBoxDom
from solver.kle_solver import KspSolver
from functions.taylor_green import velocity, vorticity, alpha
import numpy as np
import numpy.testing as np_test

import yaml
from cases.base_problem import BaseProblem 

class TestKrhs(unittest.TestCase):
    nu = 0.5/0.01
    def setUp(self):
        domain = {'nelem': [2, 2], 'lower': [0, 0], 'upper': [1, 1]}
        ngl = 9
        dm = NewBoxDom()
        dm.create(domain)
        dm.setFemIndexing(ngl)
        dm.createElement()
        mats = Matrices()
        mats.setDM(dm)
        self.dm_test = dm
        self.K_test, self.Krhs_test, self.Rw_test = mats.assembleKLEMatrices()

        with open('src/cases/taylor-green.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)

        # self.d = yamlData
        fem_ref = BaseProblem(yamlData, nelem=[2,2], ngl=ngl)
        fem_ref.setUpDomain()
        fem_ref.readMaterialData()
        fem_ref.setUpSolver()
        fem_ref.dom.computeFullCoordinates()
        # fem_ref.setUpInitialConditions()

        self.fem_ref = fem_ref

    def test_Krhs(self):
        t = 0
        nu = self.nu

        vel_ref = self.fem_ref.solverKLE.getSolution()
        vel_ref.set(0.0)
        self.fem_ref.dom.applyBoundaryConditions(vel_ref, "velocity", t, nu)

        glVec = self.dm_test.getLocalVec()
        bcs = self.dm_test.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*self.dm_test.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        freeDofs = np.array(list(set(range(glVec.getSize())) - set(list(bcDofsToSet))), dtype=np.int32)
        fullcoordArr = self.dm_test.getNodesCoordinates(indices=bcDofsToSet)
        alp = alpha(nu, t=t)
        values = velocity(fullcoordArr, alp)

        glVec.setValues(bcDofsToSet, values)
        glVec.assemble()
        # Test if both bc Velocity are equal
        np_test.assert_array_almost_equal(glVec, vel_ref, decimal=14)
        np_test.assert_array_almost_equal(glVec[freeDofs], vel_ref[freeDofs], decimal=14)
        # Test Krhs * vel
        rhs_test = self.Krhs_test * glVec
        rhs_ref = self.fem_ref.mat.Krhs * vel_ref
        lgmap = self.dm_test.getLGMap()

        rows = self.Krhs_test.getSize()[0]
        for locRow in range(rows):
            glRow = lgmap.applyInverse(locRow)[0]
            glCols, glVals = self.fem_ref.mat.Krhs.getRow(glRow)
            testCols, testValues = self.Krhs_test.getRow(locRow)
            np_test.assert_array_almost_equal(testCols, glCols, decimal=14)
            np_test.assert_array_almost_equal(testValues, glVals, decimal=14)

        np_test.assert_array_almost_equal(rhs_test, rhs_ref[freeDofs], decimal=14)

    def test_Rw(self):
        t = 0
        nu = self.nu
        dim = self.dm_test.getDimension()
        vort_ref = self.fem_ref.vort
        vort_test = self.Rw_test.createVecRight()
        allDofs = np.arange(len(vort_test.getArray())*dim, dtype=np.int32)
        coords = self.dm_test.getNodesCoordinates(indices=allDofs)
        vortValues = vorticity(coords, alpha(nu, t=t))
        vort_test.setValues(np.arange(len(vort_test.getArray()), dtype=np.int32), vortValues)
        np_test.assert_array_almost_equal(vort_test, vort_ref, decimal=14)

        rhs_ref = self.fem_ref.mat.Rw * vort_ref
        rhs_test = self.Rw_test * vort_test

        bcs = self.dm_test.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*self.dm_test.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        freeDofs = np.array(list(set(range(allDofs.shape[0])) - set(list(bcDofsToSet))), dtype=np.int32)

        np_test.assert_array_almost_equal(rhs_test, rhs_ref[freeDofs], decimal=14)

        rows = self.Rw_test.getSize()[0]
        lgmap = self.dm_test.getLGMap()

        for locRow in range(rows):
            glRow = lgmap.applyInverse(locRow)[0]
            glCols, glVals = self.fem_ref.mat.Rw.getRow(glRow)
            testCols, testValues = self.Rw_test.getRow(locRow)
            np_test.assert_array_almost_equal(testCols, glCols, decimal=14)
            np_test.assert_array_almost_equal(testValues, glVals, decimal=14)

    def test_solve(self):
        t = 0
        nu = self.nu
        dim = self.dm_test.getDimension()
        vort_test = self.Rw_test.createVecRight()
        allDofs = np.arange(len(vort_test.getArray())*dim, dtype=np.int32)
        coords = self.dm_test.getNodesCoordinates(indices=allDofs)
        vortValues = vorticity(coords, alpha(nu, t=t))
        vort_test.setValues(np.arange(len(vort_test.getArray()), dtype=np.int32), vortValues)

        glVec = self.dm_test.getLocalVec()
        bcs = self.dm_test.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*self.dm_test.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        freeDofs = np.array(list(set(range(allDofs.shape[0])) - set(list(bcDofsToSet))), dtype=np.int32)
        fullcoordArr = self.dm_test.getNodesCoordinates(indices=bcDofsToSet)
        alp = alpha(nu, t=t)
        values = velocity(fullcoordArr, alp)
        glVec.setValues(bcDofsToSet, values)
        glVec.assemble()
        rhs_test = self.Rw_test * vort_test + self.Krhs_test * glVec
        vel_ref = self.fem_ref.solverKLE.getSolution()
        vel_ref.set(0.0)
        self.fem_ref.dom.applyBoundaryConditions(vel_ref, "velocity", t, nu)
        rhs_ref = self.fem_ref.mat.Rw * vort_test + self.fem_ref.mat.Krhs * vel_ref
        np_test.assert_array_almost_equal(rhs_test, rhs_ref[freeDofs], decimal=14)
        self.fem_ref.solverKLE.solve(vort_test)
        vel_ref = self.fem_ref.solverKLE.getSolution()

        solver = KspSolver()

        solver.createSolver(self.K_test)
        solver.setRHS(self.Rw_test, self.Krhs_test)

        vel = self.dm_test.getGlobalVec()
        solver.solve(vort_test, glVec, vel)
        self.dm_test.globalToLocal(vel, glVec)
        np_test.assert_array_almost_equal(glVec[bcDofsToSet], vel_ref[bcDofsToSet], decimal=14)
        np_test.assert_array_almost_equal(glVec[freeDofs], vel_ref[freeDofs], decimal=14)

    def test_K(self):

        rows = self.K_test.getSize()[0]
        lgmap = self.dm_test.getLGMap()

        for locRow in range(rows):
            glRow = lgmap.applyInverse(locRow)[0]
            glCols, glVals = self.fem_ref.mat.K.getRow(glRow)
            testCols, testValues = self.K_test.getRow(locRow)

            locCols = lgmap.apply(glCols)

            np_test.assert_array_almost_equal(testCols, locCols, decimal=14)
            np_test.assert_array_almost_equal(testValues, glVals, decimal=14)

