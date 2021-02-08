import unittest 
from solver.ts_solver import TSSolver
import numpy as np
import numpy.testing as np_test
from petsc4py import PETSc


from matrices.operators import Operators
from functions.taylor_green import velocity, vorticity, alpha

from main import MainProblem
from domain.elements.spectral import Spectral


def rhsFunc(ts, time, vort, f, curl, srt, div, fem):
    dm = ts.getDM()
    locVort = dm.getLocalVec()
    dim = dm.getDimension()
    dim_s = 3

    # 1 Setear BC a la vorticidad.
    fem.computeBoundaryConditionsVort(time, locVort)
    # 2 Setear los valores internos a la vorticidad
    dm.globalToLocal(vort, locVort)
    # 3 resolver kle y obtener velocidad
    fem.solveKLE(locVort, time)
    vel = fem.dm.getLocalVelocity()
    # 4 aplicar VtensV
    VtensV = srt.createVecLeft()
    startInd, endInd = srt.getOwnershipRange()
    ind = np.arange(startInd, endInd, dtype=np.int32)
    arr = vel.getArray()
    v_x = arr[::dim]
    v_y = arr[1::dim]
    VtensV.setValues(ind[::dim_s], v_x**2 , False)
    VtensV.setValues(ind[1::dim_s], v_x * v_y , False)
    VtensV.setValues(ind[2::dim_s], v_y**2 , False)
    VtensV.assemble()
    # 5 Aplicar en su orden los operadores
    aux = VtensV.duplicate() 
    srt.mult(vel, aux)
    mu = 0.01
    rho = 0.5
    aux *= (2.0 * mu)
    aux.axpy(-1.0 * rho, VtensV)
    rhs = vel.duplicate()
    div.mult(aux, rhs)
    rhs.scale(1/rho)
    locF = dm.getLocalVec()
    curl.mult(rhs, locF)
    dm.localToGlobal(locF, f)
    fem.dm.restoreLocalVelocity(vel)
    dm.restoreLocalVec(locF)

def converged(ts):
    time = ts.time
    step = ts.step_number
    incr = ts.getTimeStep()
    print(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")


class TestTSSetup(unittest.TestCase):
    nu = 0.5/0.01
    nelem = [40,40]
    ngl = 3
    domain = {'nelem': nelem, 'lower': [0, 0], 'upper': [1, 1]}

    def setUp(self):
        fem  = MainProblem('taylor-green')
        fem.setUp()
        self.fem = fem
        operators = Operators()
        self.operators = operators


        dm = fem.dm.velDM
        self.operators.preallocate(self.domain, self.ngl)
        self.operators.setDM(dm)
        dim = dm.getDimension()
        elem = Spectral(self.ngl, dim)
        self.operators.setElem(elem)
        self.operators.assemble()

        ts = TSSolver()
        ts.setUpTimes(0, 1.0, 100)
        ts.setDM(self.fem.dm.vortDM)

        ts.initSolver(rhsFunc, converged, operators, fem)
        self.ts = ts

    def test_func_eval(self):
        dm = self.ts.getDM()
        self.fem.computeInitialConditions()
        globalVort = dm.getGlobalVec()
        localVort = dm.getLocalVec()
        dm.localToGlobal(localVort, globalVort)
        dm.restoreLocalVec(localVort)
        self.ts.solve(globalVort)
        raise Exception('noera')