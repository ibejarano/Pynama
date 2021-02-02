import unittest
from matrices.operators import Operators
from functions.taylor_green import velocity, vorticity, alpha
import numpy as np
import numpy.testing as np_test
from main import MainProblem
from solver.ts_solver import TSSolver


def rhsFunc(ts, time, vort, f):
    print("TS Obj", ts)
    print("time: ", time)
    print(f"Vort vec size {vort.getSize()} ", vort.getArray(True))
    print(f"F vec size {f.getSize()} ", f.getArray())
    # print(trap)
    print(" ")

def converged(ts):
    print("mas cositas convergidas")

class TestKrhs(unittest.TestCase):
    nu = 0.5/0.01
    nelem = [2,2]
    ngl = 3
    domain = {'nelem': nelem, 'lower': [0, 0], 'upper': [1, 1]}
    def setUp(self):
        fem  = MainProblem('taylor-green')
        fem.setUp()
        self.fem = fem
        operators = Operators()
        self.operators = operators
        # with open('src/cases/taylor-green.yaml') as f:
        #     yamlData = yaml.load(f, Loader=yaml.Loader)

        # # self.d = yamlData
        # fem_ref = BaseProblem(yamlData, nelem=[2,2], ngl=ngl)
        # fem_ref.setUpDomain()
        # fem_ref.readMaterialData()
        # fem_ref.setUpSolver()
        # fem_ref.dom.computeFullCoordinates()
        # # fem_ref.setUpInitialConditions()

        # self.fem_ref = fem_ref

    def test_preallocation(self):
        self.operators.preallocate(self.domain, self.ngl)

        curl = self.operators.Curl
        div = self.operators.DivSrT
        srt = self.operators.SrT

        assert curl.getSize() == (25, 50)
        assert div.getSize() == (50, 75)
        assert srt.getSize() == (75, 50)

    def test_vorticity_vec(self):
        self.operators.preallocate(self.domain, self.ngl)

        vort = self.operators.Curl.createVecLeft()

        assert vort.getSize() == 25

        vortLGMap = self.fem.dm.createVortLGMap()
        vort.setLGMap(vortLGMap)
        vort.setValuesLocal([3,4], [33, 34])
        vort.assemble()
        
        ts  = TSSolver()
        ts.setUpTimes(0, 1.0, 10)
        ts.initSolver(rhsFunc, converged)
        ts.solve(vort)
        # ts.view()

        # vort.view()
        raise Exception("bue")

    # def test_solve(self):
    #     t = 0
    #     nu = self.nu
    #     dim = self.dm_test.getDimension()
    #     vort_test = self.Rw_test.createVecRight()
    #     allDofs = np.arange(len(vort_test.getArray())*dim, dtype=np.int32)
    #     coords = self.dm_test.getNodesCoordinates(indices=allDofs)
    #     vortValues = vorticity(coords, alpha(nu, t=t))
    #     vort_test.setValues(np.arange(len(vort_test.getArray()), dtype=np.int32), vortValues)

    #     glVec = self.dm_test.getLocalVec()
    #     bcs = self.dm_test.getStratumIS("marker",1)
    #     bcDofsToSet = np.zeros(0)
    #     for poi in bcs.getIndices():
    #         arrtmp = np.arange(*self.dm_test.getPointLocal(poi)).astype(np.int32)
    #         bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

    #     freeDofs = np.array(list(set(range(allDofs.shape[0])) - set(list(bcDofsToSet))), dtype=np.int32)
    #     fullcoordArr = self.dm_test.getNodesCoordinates(indices=bcDofsToSet)
    #     alp = alpha(nu, t=t)
    #     values = velocity(fullcoordArr, alp)
    #     glVec.setValues(bcDofsToSet, values)
    #     glVec.assemble()
    #     rhs_test = self.Rw_test * vort_test + self.Krhs_test * glVec
    #     vel_ref = self.fem_ref.solverKLE.getSolution()
    #     vel_ref.set(0.0)
    #     self.fem_ref.dom.applyBoundaryConditions(vel_ref, "velocity", t, nu)
    #     rhs_ref = self.fem_ref.mat.Rw * vort_test + self.fem_ref.mat.Krhs * vel_ref
    #     np_test.assert_array_almost_equal(rhs_test, rhs_ref[freeDofs], decimal=14)
    #     self.fem_ref.solverKLE.solve(vort_test)
    #     vel_ref = self.fem_ref.solverKLE.getSolution()

    #     solver = KspSolver()

    #     solver.createSolver(self.K_test)
    #     solver.setRHS(self.Rw_test, self.Krhs_test)

    #     vel = self.dm_test.getGlobalVec()
    #     solver.solve(vort_test, glVec, vel)
    #     self.dm_test.globalToLocal(vel, glVec)
    #     np_test.assert_array_almost_equal(glVec[bcDofsToSet], vel_ref[bcDofsToSet], decimal=14)
    #     np_test.assert_array_almost_equal(glVec[freeDofs], vel_ref[freeDofs], decimal=14)
