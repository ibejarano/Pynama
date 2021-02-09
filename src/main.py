import petsc4py, sys
petsc4py.init(sys.argv)
from matrices.new_mat import Matrices
from solver.kle_solver import KspSolver
from viewer.paraviewer import Paraviewer
from functions.taylor_green import velocity, vorticity, alpha
from domain.elements.spectral import Spectral
from utils.dm_spectral import getCoordinates
from solver.ts_solver import TimeStepping

from petsc4py import PETSc
import numpy as np
import yaml
import logging

class MainProblem(object):

    def __init__(self, config, **kwargs):
        """Main class that operates the entire process of solving a problem

        Args:
            config (str or dict): str - Describing the yaml file location of the configuration
                                  dict - Describing the actual problem
        """
        try:
            with open(f'src/cases/{config}.yaml') as f:
                config = yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError:
            with open(f'{config}.yaml') as f:
                config = yaml.load(f, Loader=yaml.Loader)
        except:
            raise Exception(f"File '{config}' file not found")

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Main")
        self.logger = logger
        self.logger.info("Init problem...")
        self.opts = kwargs
        self.config = config
        self.nu = kwargs.get('nu', 0.01/0.5)

    def setUp(self):
        OptDB = PETSc.Options()
        self.dm, self.elem = self.setUpDomain(**self.opts)
        self.vel, self.vort = self.dm.createVecs()
        self.bc = None

        self.mats = Matrices()
        self.mats.setDM(self.dm.velDM)
        self.mats.setElem(self.elem)

        K, Krhs, Rw = self.assembleMatrices()

        self.solver = KspSolver()
        self.solver.createSolver(K)
        self.solver.setFromOptions()

        self.solver.setRHS(Rw, Krhs)
        self.viewer = self.setUpViewer()

        coords = self.dm.computeFullCoordinates(self.elem)
        self.coordVec = coords
        self.viewer.saveMesh(coords)

    def setUpViewer(self):
        viewer = Paraviewer()

        from datetime import datetime
        now = datetime.now()
        currDate = now.strftime("%F %H:%M:%S")
        saveDir = self.config.get("save-dir", f"{currDate}-run")

        dim = self.dm.getDimension()
        viewer.configure(dim, saveDir)

        return viewer

    def setUpDomain(self, box=True, ngl=None):
        domData = self.config.get('domain')
        if box:
            from domain.dmplex_bc import NewBoxDom as DM
            data = domData['box-mesh'] 
        else:
            from domain.dmplex_bc import NewGmshDom as DM
            data = domData['gmsh']
        dm = DM()

        dm.create(data)
        if not ngl:   
            ngl = domData['ngl']

        dm.setFemIndexing(ngl)
        dim = dm.getDimension()
        elem = Spectral(ngl, dim)
        return dm, elem
        
    def assembleMatrices(self):
        K, Krhs, Rw = self.mats.assembleKLEMatrices()
        return K, Krhs, Rw

    def solveKLE(self, vort, t=None):
        if t != None:
            self.computeBoundaryConditions(t)

        globalVel = self.dm.getGlobalVelocity()
        self.solver.solve(vort, self.vel, globalVel)
        self.dm.velDM.globalToLocal(globalVel, self.vel)
        self.dm.restoreGlobalVelocity(globalVel)

    def computeBoundaryConditions(self, t):
        dm = self.dm
        dim = dm.getDimension()
        bcs = dm.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*self.dm.velDM.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        nodesBC = [int(n/dim) for n in bcDofsToSet[::dim]]
        alp = alpha(self.nu, t=t)
        nnodes = int(self.vel.getSize()/dim)

        fullcoordArr = self.coordVec.getArray().reshape((nnodes, dim))[nodesBC]
        values = velocity(fullcoordArr, alp)

        self.vel.setValues(bcDofsToSet, values)
        self.vel.assemble()

    def computeInitialConditions(self, globalVec, t):
        print("computing initial conditions")
        alp = alpha(self.nu, t=t)
        dim = self.dm.getDimension()
        totNodes = self.vort.getSize()
        assert dim == 2
        vortValues = vorticity(self.coordVec.getArray().reshape((totNodes, dim)), alp)
        inds = np.arange(len(self.vort.getArray()), dtype=np.int32)
        self.vort.setValues(inds, vortValues)
        self.dm.vortDM.localToGlobal(self.vort, globalVec)

    def computeBoundaryConditionsVort(self, t, vort=None):
        alp = alpha(self.nu, t=t)
        dm = self.dm
        if vort == None:
            vort = self.vort
  
        dim = dm.getDimension()

        bcs = dm.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*self.dm.velDM.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        totNodes = vort.getSize()
        assert dim == 2
        nodesBC = [int(n/dim) for n in bcDofsToSet[::dim]]

        coords = self.coordVec.getArray().reshape((totNodes, dim))[nodesBC]
        vortValues = vorticity(coords , alp)
        vort.setValues(nodesBC, vortValues)
        return vort

    def computeExactVort(self, t):
        alp = alpha(self.nu, t=t)
        dm = self.dm
        vort = dm.getLocalVorticity()
        dim = self.dm.getDimension()
        totNodes = vort.getSize()
        assert dim == 2
        vortValues = vorticity(self.coordVec.getArray().reshape((totNodes, dim)), alp)
        inds = np.arange(len(vort.getArray()), dtype=np.int32)
        vort.setValues(inds, vortValues)
        return vort

    def destroy(self):
        self.dm.destroy()
        self.solver.destroy()

    def saveStep(self, ts=None ,step=None, time=None):
        if ts:
            time = ts.time
            step = ts.step_number
            incr = ts.getTimeStep()
            print(f"TS Converged :  Step: {step:6} | Time {time:5.4f} | Delta: {incr:.4f} ")
        else:
            print(f"Saving Step {step:6} | Time {time:5.4f} ")

        self.viewer.saveData(step, time, self.vel)
        self.viewer.writeXmf("TS-Solver-TG-testing")

class TestingFem(MainProblem):
    def __init__(self, nelem, **kwargs):
        self.config = dict()
        boxMesh = {"nelem": nelem, "lower": [0,0], "upper":[1,1]}
        self.config['domain'] = {"box-mesh": boxMesh}
        self.opts = kwargs
        self.nu = kwargs.get('nu', 1)

    def getVelocityErrorNorm(self, t):
        errVec = self.getErrorVec(t)
        errNorm = errVec.norm(norm_type=2)
        errVec.destroy()
        return errNorm

    def getErrorVec(self, t):
        alp = alpha(self.nu, t)
        locVel = self.dm.getLocalVelocity()
        exactVel = locVel.duplicate()
        exactVel.setName('exact-vel')

        nnodes = int(locVel.getSize()/self.dm.getDimension())
        
        fullcoordArr = self.coordVec.getArray().reshape((nnodes, 2))
        values = velocity(fullcoordArr, alp)
        exactVel.setValues(np.arange(len(exactVel.getArray()), dtype=np.int32), values)

        err = (exactVel-locVel)
        err.setName('error')
        exactVel.destroy()
        self.dm.restoreLocalVelocity(locVel)
        return err

    def saveStep(self, step, time):
        err = self.getErrorVec(time)
        vel = self.dm.getLocalVelocity()
        self.viewer.saveData(step, time,  err, vel)
        self.dm.restoreLocalVelocity(vel)

    def setUpViewer(self):
        viewer = Paraviewer()

        from datetime import datetime
        now = datetime.now()
        currDate = now.strftime("%F")
        saveDir =  f"{currDate}-testing"

        dim = self.dm.getDimension()
        viewer.configure(dim, saveDir)

        return viewer

if __name__ == "__main__":
    opts = PETSc.Options()
    runTest = opts.getString('test', False)

    if not runTest:
        fem = MainProblem('taylor-green')
        fem.setUp()

        ts = TimeStepping()
        ts.setFem(fem)
        ts.setUp()
        ts.startSolver()
        # t = 0.0
        # vort = fem.computeExactVort(t)
        # fem.solveKLE(vort, t)
        # fem.saveStep(step=1, time=t)
        print("Time stepping Finished")

    # Convergence test
    if runTest == 'kle':
        nglStart = opts.getInt('nglStart', 3)
        nglEnd = opts.getInt('nglEnd', 15)
        fileOut = opts.getString('file-out', f"kle-convergence-ngl{nglStart}to{nglEnd}" )
        ngls=list()
        timeComputed=list()
        errors= list()
        taus = list()
        nu = 1
        viscousTimes = [0, 0.05, 0.2, 0.5, 0.75, 0.9]
        times = [(tau**2)/(4*nu) for tau in viscousTimes]

        print("Running convergence KLE Test from ngl: {nglStart} to {nglEnd}")

        for ngl in range(nglStart,nglEnd):
            print("Running test with ngl:", ngl)
            testFem = TestingFem([2,2], ngl=ngl)
            testFem.setUp()

            for i, t in enumerate(times):
                vort = testFem.computeExactVort(t)
                testFem.solveKLE(vort, t)
                err = testFem.getErrorVec(t).norm(norm_type=2)
                
                ngls.append(ngl)
                errors.append(err)
                timeComputed.append(float(t))
                taus.append(float(viscousTimes[i]))

            testFem.destroy()

        output = {"ngl": ngls, "errors": errors, "times": timeComputed, "vTimes": taus  }
        with open(f"{fileOut}.yaml", 'w') as f:
            yaml.dump(output, f)

        print(f"Test finished. File {fileOut}.yaml created")

    if runTest == 'viewError':
        fem = TestingFem([5,5], ngl=5)
        fem.setUp()
        
        viscousTimes = [0, 0.05, 0.1 , 0.2, 0.5, 0.75, 0.9]
        times = [(tau**2)/(4*0.5/0.01) for tau in viscousTimes]
        
        for step, t in enumerate(times):

            vort = fem.computeExactVort(t)
            fem.solveKLE(vort, t)
            fem.saveStep(step=step, time=t)

        fem.viewer.writeXmf("TG-testing")

        print("Finished")