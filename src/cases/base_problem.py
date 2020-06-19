import yaml
from petsc4py import PETSc
from mpi4py import MPI
# Local packages
from domain.dmplex import DMPlexDom
from elements.spectral import Spectral
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver
from matrices.mat_generator import Mat
from matrices.mat_ns import MatNS
from solver.ksp_solver import KspSolver
import logging
import numpy as np

class BaseProblem(object):
    def __init__(self, comm=MPI.COMM_WORLD, **kwargs):
        """
        comm: MPI Communicator
        """
        self.comm = comm
        self.logger = logging.getLogger("")
        case = PETSc.Options().getString('case')
        try:
            with open(f'src/cases/{case}.yaml') as f:
                yamlData = yaml.load(f, Loader=yaml.Loader)
            self.logger.info(f"Initializing problem: {yamlData['name']}")
        except:
            self.logger.info(f"Case '{case}' Not Found")

        for key in kwargs.keys():
            try:
                assert  yamlData['domain'][key]
                yamlData['domain'][key] = kwargs[key]
            except:
                print(f"Key >> {key} << not defined in yaml")

        self.caseName = yamlData['name']
        self.readInputData(yamlData['domain'])
      
        if 'time-solver' in yamlData:
            self.setUpTimeSolver(yamlData['time-solver'])
        if 'boundary-conditions' in yamlData:
            self.readBoundaryCondition(yamlData['boundary-conditions'])

    def setUpDomain(self):
        self.dom = DMPlexDom(self.lower, self.upper, self.nelem)
        self.logger.debug("DMPlex dom intialized")

    def setUpHighOrderIndexing(self):
        self.dom.setFemIndexing(self.ngl)

    def setUpElement(self):
        self.elemType = Spectral(self.ngl, self.dim)

    def setUpBoundaryConditions(self):
        self.dom.setLabelToBorders()
        self.tag2BCdict, self.node2tagdict = self.dom.setBoundaryCondition()


    def readInputData(self, inputData):
        self.dim = len(inputData['nelem'])
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        self.ngl = inputData['ngl']
        self.lower = inputData['lower']
        self.upper = inputData['upper']
        self.nelem = inputData['nelem']

    def createMesh(self):
        self.dom.computeFullCoordinates(self.elemType)
        self.viewer = Paraviewer(self.dim ,self.comm)
        self.viewer.saveMesh(self.dom.fullCoordVec)

    def setUpGeneral(self):
        self.setUpDomain()
        self.setUpHighOrderIndexing()
        self.setUpElement()
        self.createMesh()

    def buildOperators(self):
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locSrT, locDivSrT, locCurl, locWei = self.elemType.getElemKLEOperators(cornerCoords)
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=False)
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
            indicesSrT = self.dom.getSrtIndex(nodes)
            self.mat.SrT.setValues(indicesSrT, indicesVel, locSrT, True)
            self.mat.DivSrT.setValues(indicesVel, indicesSrT, locDivSrT, True)
            self.mat.Curl.setValues(indicesW, indicesVel, locCurl, True)

            self.mat.weigSrT.setValues(indicesSrT, np.repeat(locWei, self.dim_s), True)
            self.mat.weigDivSrT.setValues(indicesVel, np.repeat(locWei, self.dim), True)
            self.mat.weigCurl.setValues(indicesW, np.repeat(locWei, self.dim_w), True)

        self.mat.assembleOperators()

    def getTS(self):
        ts = TsSolver(self.comm)
        return ts

    def setUpTimeSolver(self, inputData):
        self.ts = self.getTS()
        sTime = inputData['start-time']
        eTime = inputData['end-time']
        maxSteps = inputData['max-steps']
        self.ts.setUpTimes(sTime, eTime, maxSteps)
        self.ts.initSolver(self.evalRHS, self.convergedStepFunction)

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        #vort = ts.getSolution()
        self.logger.info(f"Converged: Step {step} Time {time}")
        # lo de abajo en otro lado
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort])

    def getBoundaryNodes(self):
        """ IS: Index Set """
        nodesSet = set()
        IS =self.dom.getStratumIS('marco', 0)
        entidades = IS.getIndices()
        for entity in entidades:
            nodes = self.dom.getGlobalNodesFromCell(entity, False)
            nodesSet |= set(nodes)
        return list(nodesSet)

    def evalRHS(self, ts, t, Vort, f):
        """Evaluate the KLE right hand side."""
        # KLE spatial solution with vorticity given
        rho = 1
        mu = 1
        self.solveKLE(t, Vort)

        # FIXME: Generalize for dim = 3 also
        sK, eK = self.mat.K.getOwnershipRange()

        for indN in range(sK, eK, self.dim):
            indicesVV = [indN * self.dim_s / self.dim + d
                         for d in range(self.dim_s)]
            VelN = self.vel.getValues([indN + d for d in range(self.dim)])
            if self.dim==2:
                VValues = [VelN[0] ** 2, VelN[0] * VelN[1], VelN[1] ** 2]
            elif self.dim==3:
                VValues = [VelN[0] ** 2, VelN[0] * VelN[1] ,VelN[1] ** 2 , VelN[1] * VelN[2] , VelN[2] **2 , VelN[2] *VelN[0]]
            else:
                raise Exception("Wrong dim")

            self._VtensV.setValues(indicesVV, VValues, False)

        self._VtensV.assemble()

        # self._Aux1 = self.SrT * self._Vel
        self.mat.SrT.mult(self.vel, self._Aux1)

        # _Aux1 = 2*Mu * S - rho * Vvec ^ VVec
        self._Aux1 *= (2.0 * mu)
        self._Aux1.axpy(-1.0 * rho, self._VtensV)

        # FIXME: rhs should be created previously or not?
        rhs = self.vel.duplicate()
        # RHS = Curl * Div(SrT) * 2*Mu * S - rho * Vvec ^ VVec
            # rhs = (self.DivSrT * self._Aux1) / self.rho
        self.mat.DivSrT.mult(self._Aux1, rhs)
        rhs.scale(1/rho)

        self.mat.Curl.mult(rhs, f)

    def startSolver(self):
        self.computeInitialCondition(startTime = 0.0)
        self.ts.solve(self.vort)

    def solveKLE(self, time, vort):
        pass

    def buildKLEMats(self):
        pass

    def computeInitialCondition(self, startTime):
        pass

    def applyBoundaryConditions(self, time, bcNodes):
        pass

class NoSlip(BaseProblem):
    
    def setUpEmptyMats(self):
        self.mat = MatNS(self.dim, self.comm)
        fakeConectMat = self.dom.getDMConectivityMat()
        globalIndicesNS = self.dom.getGlobalIndicesDirichlet() # TODO Reuso todos los indices dirichlet, se podria cambiar el nombre
        self.mat.createEmptyKLEMats(fakeConectMat, globalIndicesNS, createOperators=True)

    def setUpSolver(self):
        self.solver = KspSolver()
        self.solverFS = KspSolver()
        self.solverFS.createSolver(self.mat.K + self.mat.Kfs, self.comm) 
        self.solver.createSolver(self.mat.K , self.comm)
        self.vel = self.mat.K.createVecRight()
        self.vel.setName("velocity")
        self.vort = self.mat.Rw.createVecRight()
        self.vort.setName("vorticity")
        self.vort.set(0.0)

        sK, eK = self.mat.K.getOwnershipRange()
        locRowsK = eK - sK

        self._VtensV = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)
        self._Aux1 = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)

    def solveKLE(self, time, vort):
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        self.velFS = self.vel.copy()
        self.solverFS( self.mat.Rw * vort + self.mat.Rwfs * vort\
             + self.mat.Krhs * self.vel , self.velFS)
        self.applyBoundaryConditionsFS(time, boundaryNodes)
        vort= self.mat.Curl *self.velFS
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def readBoundaryCondition(self,inputData):
        bcdict = inputData['border-name']
        self.BoundaryCondition=[]
        for bc in bcdict.keys():
            if bc[:5]=="upper":
                self.BoundaryCondition.append((self.upper,bcdict[bc]["coord"],bcdict[bc]["vel"]))
            if bc[:5]=="lower":
                self.BoundaryCondition.append((self.lower,bcdict[bc]["coord"],bcdict[bc]["vel"]))


    def buildKLEMats(self):
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        boundaryNodes = set(self.getBoundaryNodes())
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)
        indices2onefs = set()
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            self.logger.debug("DMPlex cell: %s", cell)

            nodes = self.dom.getGlobalNodesFromCell(cell, shared=False)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
            nodeBCintersect = boundaryNodes & set(nodes)
            # self.logger.debug("te intersecto: %s", nodeBCintersect)
            
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions
 
            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                nsNorm = set()
                coord=self.dom.getNodesCoordinates([node])[0]
                for i in range(self.dim):
                    if (coord[i]==self.upper[i]) or (coord[i]==self.lower[i]):
                        nsNorm.add(i)
                dofSetFSNS.update([localBoundaryNode*self.dim + d
                                       for d in nsNorm])
                dofFreeFSSetNS.update([localBoundaryNode*self.dim + d
                                        for d in (set(range(self.dim)) - nsNorm)])


            dofFree = list(set(range(len(indicesVel)))
                           - dofFreeFSSetNS - dofSetFSNS)
            dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
            dofFreeFSSetNS = list(dofFreeFSSetNS)
            dofSetFSNS = list(dofSetFSNS)
            gldofFreeFSSetNS = [indicesVel[ii] for ii in dofFreeFSSetNS]
            gldofSetFSNS = [indicesVel[ii] for ii in dofSetFSNS]
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            

            if nodeBCintersect:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)
                self.mat.Kfs.setValues(gldofFreeFSSetNS, gldofFree,
                                    locK[np.ix_(dofFreeFSSetNS, dofFree)],
                                    addv=True)

                self.mat.Kfs.setValues(gldofFree, gldofFreeFSSetNS,
                                    locK[np.ix_(dofFree, dofFreeFSSetNS)],
                                    addv=True)

                self.mat.Kfs.setValues(
                    gldofFreeFSSetNS, gldofFreeFSSetNS,
                    locK[np.ix_(dofFreeFSSetNS, dofFreeFSSetNS)],
                    addv=True)

                # Indices where diagonal entries should be reduced by 1
                indices2onefs.update(gldofFreeFSSetNS)

                self.mat.Rwfs.setValues(gldofFreeFSSetNS, indicesW,
                                    locRw[dofFreeFSSetNS, :], addv=True)

                self.mat.Rdfs.setValues(gldofFreeFSSetNS, nodes,
                                    locRd[dofFreeFSSetNS, :], addv=True)
                self.mat.Krhsfs.setValues(
                        gldofFreeFSSetNS, gldofSetFSNS,
                        - locK[np.ix_(dofFreeFSSetNS, dofSetFSNS)], addv=True)
                self.mat.Krhsfs.setValues(
                        gldofFree, gldofSetFSNS,
                        - locK[np.ix_(dofFree, dofSetFSNS)], addv=True)
                for indd in gldofSetFSNS:
                        self.mat.Krhsfs.setValues(indd, indd, 0, addv=True)


            self.mat.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.mat.K.setValues(indd, indd, 0, addv=True)

            self.mat.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.mat.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)

        self.mat.assembleAll()
        self.mat.setIndices2One(indices2one)

        for indd in indices2onefs:
            self.mat.Kfs.setValues(indd, indd, -1, addv=True)

        self.mat.Kfs.assemble()
        self.mat.Rwfs.assemble()
        self.mat.Rdfs.assemble()
        self.mat.Krhsfs.assemble()

        for indd in (indices2one - indices2onefs):
            self.mat.Krhsfs.setValues(indd, indd, 1, addv=False)
        self.mat.Krhsfs.assemble()

class FreeSlip(BaseProblem):

    def generateExactVecs(self, time):
        return 0, 0

    def buildMatrices(self):
        pass

    def setUpSolver(self):
        self.solver = KspSolver()
        self.solver.createSolver(self.mat.K, self.comm)
        self.vel = self.mat.K.createVecRight()
        self.vel.setName("velocity")
        self.vort = self.mat.Rw.createVecRight()
        self.vort.setName("vorticity")
        self.vort.set(0.0)

        sK, eK = self.mat.K.getOwnershipRange()
        locRowsK = eK - sK

        self._VtensV = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)
        self._Aux1 = PETSc.Vec().createMPI(
            ((locRowsK * self.dim_s / self.dim, None)), comm=self.comm)

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        fakeConectMat = self.dom.getDMConectivityMat()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()
        self.mat.createEmptyKLEMats(fakeConectMat, globalIndicesDIR, createOperators=True)

    def solveKLE(self, time, vort):
        boundaryNodes = self.getBoundaryNodes()
        self.applyBoundaryConditions(time, boundaryNodes)
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def getKLEError(self, times=None ,startTime=0.0, endTime=1.0, steps=10):
        try:
            assert times !=None
        except:
            times = np.arange(startTime, endTime, (endTime - startTime)/steps)

        boundaryNodes = self.getBoundaryNodes()
        errors = list()
        for time in times:
            exactVel, exactVort = self.generateExactVecs(time)
            self.applyBoundaryConditions(time, boundaryNodes)
            self.solver( self.mat.Rw * exactVort + self.mat.Krhs * self.vel , self.vel)
            error = (exactVel - self.vel).norm(norm_type=2)
            errors.append(error)
        return errors

    def readBoundaryCondition(self,inputData):
        pass

    def buildKLEMats(self):
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        boundaryNodes = set(self.node2tagdict.keys())
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)

        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            self.logger.debug("DMPlex cell: %s", cell)

            nodes = self.dom.getGlobalNodesFromCell(cell, shared=False)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
            nodeBCintersect = boundaryNodes & set(nodes)
            # self.logger.debug("te intersecto: %s", nodeBCintersect)
            
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions

            for node in nodeBCintersect:
                localBoundaryNode = nodes.index(node)
                # FIXME : No importa el bc, #TODO cuando agregemos NS si importa
                for dof in range(self.dim):
                    dofSetFSNS.add(localBoundaryNode*self.dim + dof)

            dofFree = list(set(range(len(indicesVel)))
                           - dofFreeFSSetNS - dofSetFSNS)
            dof2beSet = list(dofFreeFSSetNS | dofSetFSNS)
            dofFreeFSSetNS = list(dofFreeFSSetNS)
            dofSetFSNS = list(dofSetFSNS)
            gldof2beSet = [indicesVel[ii] for ii in dof2beSet]
            gldofFree = [indicesVel[ii] for ii in dofFree]
            

            if nodeBCintersect:
                self.mat.Krhs.setValues(
                gldofFree, gldof2beSet,
                -locK[np.ix_(dofFree, dof2beSet)], addv=True)
                indices2one.update(gldof2beSet)

                # FIXME: is the code below really necessary?
                for indd in gldof2beSet:
                    self.mat.Krhs.setValues(indd, indd, 0, addv=True)

            self.mat.K.setValues(gldofFree, gldofFree,
                             locK[np.ix_(dofFree, dofFree)], addv=True)

            for indd in gldof2beSet:
                self.mat.K.setValues(indd, indd, 0, addv=True)

            self.mat.Rw.setValues(gldofFree, indicesW,
                              locRw[np.ix_(dofFree, range(len(indicesW)))], addv=True)

            self.mat.Rd.setValues(gldofFree, nodes,
                              locRd[np.ix_(dofFree, range(len(nodes)))],
                              addv=True)
        
        self.mat.assembleAll()
        self.mat.setIndices2One(indices2one)