import petsc4py, sys
petsc4py.init(sys.argv)
from matrices.new_mat import Matrices
from solver.kle_solver import KspSolver
from viewer.paraviewer import Paraviewer
from functions.taylor_green import velocity, vorticity, alpha


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
            self.logger.info(f"File '{config}' file not found")


        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Main")
        self.logger = logger
        self.logger.info("Init problem...")
        self.opts = kwargs
        # TODO setFromOptions
        self.validate(config)
        self.config = config
        self.nu = kwargs.get('nu', 1)

    def validate(self, configData):
        requiredData = ("domain", "boundary-conditions", "material-properties")
        missingKeys = list()
        for required in requiredData:
            if required not in configData.keys():
                missingKeys.append(required)
        if len(missingKeys):
            raise Exception(f"The following key(s) MUST be defined: {missingKeys} ")

    def validateDomain(self, domainData: dict):
        ngl = domainData.get('ngl', False)
        if not ngl:
            raise Exception("NGL not defined")
        if 'box-mesh' in domainData:
            boxMesh = domainData['box-mesh']
            lower = boxMesh.get('lower', [0,0])
            upper = boxMesh.get('upper', [1,1])
            nelem = boxMesh.get('nelem', False)
            if not nelem:
                raise Exception("Number of elements not defined in boxmesh (Key 'nelem')")
            # Create boxmesh with the data from that place
        elif 'gmsh' in domainData:
            fileLocation = domainData['file']
            raise Exception("Not implemented yet")
        else:
            raise Exception("Domain must incluide 'box-mesh' data or 'gmsh' file location")

    def validateBoundaryConditions(self, bcData: dict):
        if 'custom-func' in bcData:
            pass
        elif 'uniform' in bcData:
            pass
        elif ('free-slip' in bcData) and ('no-slip' in bcData):
            pass
        elif 'free-slip' in bcData:
            pass
        elif 'no-slip' in bcData:
            pass
        else:
            raise Exception("Wrong boundary conditions")

    def setUp(self):
        OptDB = PETSc.Options()
        self.dm = self.setUpDomain(**self.opts)
        self.bc = None

        self.mats = Matrices()
        self.mats.setDM(self.dm)
        K, Krhs, Rw = self.assembleMatrices()

        self.solver = KspSolver()
        self.solver.createSolver(K)
        self.solver.setFromOptions()

        vort = Rw.createVecRight()
        vort.setName("vorticity")
        vel = self.dm.getLocalVec()
        vel.setName("velocity")
        self.dm.restoreLocalVec(vel)

        self.vort = vort

        self.solver.setRHS(Rw, Krhs)
        self.viewer = self.setUpViewer()
        self.viewer.saveMesh(self.dm.fullCoordVec)

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
        if box:
            from domain.dmplex_bc import NewBoxDom as DM
            data = self.config['domain']['box-mesh'] 
        else:
            from domain.dmplex_bc import NewGmshDom as DM
            data = self.config['domain']['gmsh']
        dm = DM()
        dm.create(data)
        if ngl:
            dm.setFemIndexing(ngl)
        else:    
            dm.setFemIndexing(self.config['domain']['ngl'])
        dm.createElement()
        return dm
        
    def assembleMatrices(self):
        K, Krhs, Rw = self.mats.assembleKLEMatrices()
        return K, Krhs, Rw

    def solveKLE(self, vort, t=None):
        if t != None:
            # print(f"Setting time-dependant BC t = {t}")
            self.computeBoundaryConditions(t)

        vel = self.dm.getGlobalVec()
        glVec = self.dm.getLocalVec()
        self.solver.solve(vort, glVec, vel)
        self.dm.globalToLocal(vel, glVec)
        self.dm.restoreLocalVec(glVec)

    def computeBoundaryConditions(self, t):
        dm = self.dm
        glVec = dm.getLocalVec()
        bcs = dm.getStratumIS("marker",1)
        bcDofsToSet = np.zeros(0)
        for poi in bcs.getIndices():
            arrtmp = np.arange(*dm.getPointLocal(poi)).astype(np.int32)
            bcDofsToSet = np.append(bcDofsToSet, arrtmp).astype(np.int32)

        fullcoordArr = dm.getNodesCoordinates(indices=bcDofsToSet)
        alp = alpha(self.nu, t=t)
        values = velocity(fullcoordArr, alp)
        glVec.setValues(bcDofsToSet, values)
        glVec.assemble()
        self.dm.restoreLocalVec(glVec)

    def computeExactVort(self, t):
        alp = alpha(self.nu, t=t)
        dm = self.dm
        vort = self.vort
        dim = self.dm.getDimension()
        allDofs = np.arange(len(vort.getArray())*dim, dtype=np.int32)
        coords = dm.getNodesCoordinates(indices=allDofs)
        vortValues = vorticity(coords, alp)
        vort.setValues(np.arange(len(vort.getArray()), dtype=np.int32), vortValues)
        return vort

    def destroy(self):
        self.dm.destroy()
        self.vort.destroy()
        self.solver.destroy()

    def saveStep(self, step, time):
        globalVel = self.dm.getLocalVec()
        self.viewer.saveData(step, time, globalVel)
        self.viewer.writeXmf("TG-testing")
        self.dm.restoreLocalVec()


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
        glVec = self.dm.getLocalVec()
        exactVel = glVec.duplicate()
        exactVel.setName('exact-vel')
        allDofs = np.arange(len(exactVel.getArray()), dtype=np.int32)
        coords = self.dm.getNodesCoordinates(indices=allDofs)
        values = velocity(coords, alp)
        exactVel.setValues(np.arange(len(exactVel.getArray()), dtype=np.int32), values)

        err = (exactVel-glVec)
        err.setName('error')
        exactVel.destroy()
        self.dm.restoreLocalVec(glVec)
        return err

    def saveStep(self, step, time):
        err = self.getErrorVec(time)
        globalVel = self.dm.getLocalVec()
        self.viewer.saveData(step, time,  err, globalVel)


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
        t = 0.0
        vort = fem.computeExactVort(t)
        fem.solveKLE(vort, t)
        fem.saveStep(1, t)
        print("Finished")


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
        fem = TestingFem([2,2], ngl=4)
        fem.setUp()
        
        
        viscousTimes = [0, 0.05, 0.1 , 0.2, 0.5, 0.75, 0.9]
        times = [(tau**2)/(4*0.5/0.01) for tau in viscousTimes]
        
        for step, t in enumerate(times):

            vort = fem.computeExactVort(t)
            fem.solveKLE(vort, t)
            fem.saveStep(step, t)

        fem.viewer.writeXmf("TG-testing")

        print("Finished")