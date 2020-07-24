import yaml
from petsc4py import PETSc
from mpi4py import MPI
# Local packages
from domain.dmplex import DMPlexDom
from elements.spectral import Spectral
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver
from matrices.mat_generator import Mat, Operators
from matrices.mat_ns import MatNS
from solver.ksp_solver import KspSolver
from common.timer import Timer
import logging
import numpy as np

class BaseProblem(object):
    def __init__(self, comm=MPI.COMM_WORLD, **kwargs):
        """
        comm: MPI Communicator
        """
        self.comm = comm
        self.timerTotal= Timer()
        self.timerTotal.tic()
        self.timer = Timer()
        try:
            case = kwargs['case']
        except:
            case = PETSc.Options().getString('case', 'uniform' )
        try:
            with open(f'src/cases/{case}.yaml') as f:
                yamlData = yaml.load(f, Loader=yaml.Loader)
            self.logger = logging.getLogger(yamlData['name'])
            if not self.comm.rank:
                self.logger.info("Initializing problem...")
        except:
            self.logger.info(f"Case '{case}' Not Found")

        self.caseName = yamlData['name']
        self.readDomainData(yamlData['domain'])
        self.readMaterialData(yamlData['material-properties'])
        if 'time-solver' in yamlData:
            self.setUpTimeSolver(yamlData['time-solver'])
        if 'boundary-conditions' in yamlData:
            self.readBoundaryCondition(yamlData['boundary-conditions'])

    def setUp(self):
        self.setUpGeneral()
        self.setUpBoundaryConditions()
        self.setUpEmptyMats()
        self.buildKLEMats()
        self.buildOperators()

    def setUpDomain(self):
        self.dom = DMPlexDom(self.lower, self.upper, self.nelem)
        self.dom.setFemIndexing(self.ngl)
        if not self.comm.rank:
            self.logger.info(f"DMPlex dom created")

    def setUpElement(self):
        self.elemType = Spectral(self.ngl, self.dim)
        if not self.comm.rank:
            self.logger.info(f"{self.dim}-D ngl:{self.ngl} Spectral element created")

    def setUpBoundaryConditions(self):
        self.dom.setLabelToBorders()
        self.dom.setBoundaryCondition()
        if not self.comm.rank:
            self.logger.info(f"Boundary Conditions setted up")

    def readMaterialData(self, inputData):
        self.rho = inputData['rho']
        self.mu = inputData['mu']

    def readDomainData(self, inputData):
        self.dim = len(inputData['nelem'])
        self.dim_w = 1 if self.dim == 2 else 3
        self.dim_s = 3 if self.dim == 2 else 6
        self.ngl = inputData['ngl']
        self.lower = inputData['lower']
        self.upper = inputData['upper']
        self.nelem = inputData['nelem']

    def createMesh(self):
        self.viewer = Paraviewer(self.dim ,self.comm)
        self.dom.computeFullCoordinates(self.elemType)
        self.viewer.saveMesh(self.dom.fullCoordVec)
        if not self.comm.rank:
            self.logger.info(f"Mesh of {self.nelem} created")

    def setUpGeneral(self):
        self.setUpDomain()
        self.setUpElement()
        self.createMesh()

    # @profile
    def buildOperators(self):
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        localOperators = self.elemType.getElemKLEOperators(cornerCoords)
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            self.operator.setValues(localOperators, nodes)
        self.operator.assembleAll()
        if not self.comm.rank:
            self.logger.info(f"Operators Matrices builded")

    def setUpTimeSolver(self, inputData):
        self.ts = TsSolver(self.comm)
        sTime = inputData['start-time']
        eTime = inputData['end-time']
        maxSteps = inputData['max-steps']
        self.ts.setUpTimes(sTime, eTime, maxSteps)
        self.ts.initSolver(self.evalRHS, self.convergedStepFunction)

    def convergedStepFunction(self, ts):
        time = ts.time
        step = ts.step_number
        incr = ts.getTimeStep()
        proc = self.vort.copy()
        proc.setName("num proc")
        # procSize = proc.getLocalSize()
        # print(f"{proc.getOwnershipRange() = }")
        beg, end = proc.getOwnershipRange() 
        # locSize = proc.getLocalSize()
        # dofs = list(range(locSize))
        # print(len(dofs), dofs)
        # print(locSize)
        for i in range(beg,end):
            proc.setValue(i, self.comm.rank)
        proc.assemble()
        # proc.setValuesLocal(dofs, [self.comm.rank]*locSize)
        # proc.assemble()
        self.viewer.saveVec(self.vel, timeStep=step)
        self.viewer.saveVec(self.vort, timeStep=step)
        self.viewer.saveVec(proc, timeStep=step)
        self.viewer.saveStepInXML(step, time, vecs=[self.vel, self.vort, proc])
        self.createVtkFile()

        if not self.comm.rank:
            self.logger.info(f"Converged: Step {step:4} | Time {time:.4e} | Increment Time: {incr:.2e} ")

    def createVtkFile(self):
        viewer = PETSc.Viewer()
        viewer.createVTK('immersed-body.vtk', mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(self.dom)
        viewer.destroy()

    def getBoundaryNodes(self):
        """ IS: Index Set """
        nodesSet = set()
        IS =self.dom.getStratumIS('marco', 0)
        entidades = IS.getIndices()
        for entity in entidades:
            nodes = self.dom.getGlobalNodesFromCell(entity, False)
            nodesSet |= set(nodes)
        return list(nodesSet)

    # @profile
    def evalRHS(self, ts, t, Vort, f):
        """Evaluate the KLE right hand side."""
        # KLE spatial solution with vorticity given
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
        self.operator.SrT.mult(self.vel, self._Aux1)

        # _Aux1 = 2*Mu * S - rho * Vvec ^ VVec
        self._Aux1 *= (2.0 * self.mu)
        self._Aux1.axpy(-1.0 * self.rho, self._VtensV)

        # FIXME: rhs should be created previously or not?
        rhs = self.vel.duplicate()
        # RHS = Curl * Div(SrT) * 2*Mu * S - rho * Vvec ^ VVec
            # rhs = (self.DivSrT * self._Aux1) / self.rho
        self.operator.DivSrT.mult(self._Aux1, rhs)
        rhs.scale(1/self.rho)

        self.operator.Curl.mult(rhs, f)

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

    def readBoundaryCondition(self,inputData):
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
        self.mat = None
        self.operator = None

class NoSlip(BaseProblem):
    def setUpEmptyMats(self):
        self.mat = MatNS(self.dim, self.comm)
        self.operator = Operators(self.dim, self.comm)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()

        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, globalIndicesDIR)
        if not self.comm.rank:
            self.logger.info(f"Empty KLE Matrices created")
        
        self.operator.createAll(rStart, rEnd, d_nnz_ind, o_nnz_ind)
        if not self.comm.rank:
            self.logger.info(f"Empty Operators created")

    def setUpSolver(self):
        super().setUpSolver()
        self.solverFS = KspSolver()
        self.solverFS.createSolver(self.mat.K + self.mat.Kfs, self.comm) 
        self.velFS = self.vel.copy()

    def solveKLE(self, time, vort):
        self.applyBoundaryConditions()
        self.solverFS( self.mat.Rw * vort + self.mat.Rwfs * vort\
             + self.mat.Krhs * self.vel , self.velFS)
        self.applyBoundaryConditionsFS()
        vort = self.operator.Curl * self.velFS
        self.solver( self.mat.Rw * vort + self.mat.Krhs * self.vel , self.vel)

    def buildKLEMats(self):
        indices2one = set()  # matrix indices to be set to 1 for BC imposition
        boundaryNodes = self.mat.globalIndicesNS
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, locRd = self.elemType.getElemKLEMatrices(cornerCoords)
        indices2onefs = set()
        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
            nodeBCintersect = boundaryNodes & set(nodes)
            dofFreeFSSetNS = set()  # local dof list free at FS sol
            dofSetFSNS = set()  # local dof list set at both solutions
            # for node in nodeBCintersect:
            #     localBoundaryNode = nodes.index(node)
            #     nsNorm = set()
            #     coord=self.dom.getNodesCoordinates([node])[0]
            #     for i in range(self.dim):
            #         if (coord[i]==self.upper[i]) or (coord[i]==self.lower[i]):
            #             nsNorm.add(i)
            #     dofSetFSNS.update([localBoundaryNode*self.dim + d
            #                            for d in nsNorm])
            #     dofFreeFSSetNS.update([localBoundaryNode*self.dim + d
            #                             for d in (set(range(self.dim)) - nsNorm)])

            if nodeBCintersect:
                borderNodes, normals = self.dom.getBorderNodesWithNormal(cell)
                for i, globBorderNodes in enumerate(borderNodes):
                    tangentialDofs = list(range(self.dim))
                    tangentialDofs.pop(normals[i])
                    localNodes = [ nodes.index(node) for node in globBorderNodes ]
                    normalDof = normals[i]
                    # print(f"{[self.comm.rank]} {nodes = } {globBorderNodes = } {normals = } {localNodes = } ")
                    dofSetFSNS.update( [locNode * self.dim + normalDof for locNode in localNodes ]  )
                    dofFreeFSSetNS.update( [locNode * self.dim + dof for locNode in localNodes for dof in tangentialDofs] )
                dofFreeFSSetNS -= dofSetFSNS

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

        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")

class FreeSlip(BaseProblem):
    def generateExactVecs(self, time):
        return 0, 0

    def buildMatrices(self):
        pass

    def setUpEmptyMats(self):
        self.mat = Mat(self.dim, self.comm)
        self.operator = Operators(self.dim, self.comm)
        rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o = self.dom.getMatIndices()
        globalIndicesDIR = self.dom.getGlobalIndicesDirichlet()

        self.mat.createEmptyKLEMats(rStart, rEnd, d_nnz_ind, o_nnz_ind, ind_d, ind_o, globalIndicesDIR)
        if not self.comm.rank:
            self.logger.info(f"Empty KLE Matrices created")

        self.operator.createAll(rStart, rEnd, d_nnz_ind, o_nnz_ind)
        if not self.comm.rank:
            self.logger.info(f"Empty Operators created")

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

    def buildKLEMats(self):
        indices2one = set() 
        boundaryNodes = self.mat.globalIndicesDIR 
        cornerCoords = self.dom.getCellCornersCoords(cell=0)
        locK, locRw, _ = self.elemType.getElemKLEMatrices(cornerCoords)

        for cell in range(self.dom.cellStart, self.dom.cellEnd):
            nodes = self.dom.getGlobalNodesFromCell(cell, shared=True)
            # Build velocity and vorticity DoF indices
            indicesVel = self.dom.getVelocityIndex(nodes)
            indicesW = self.dom.getVorticityIndex(nodes)
           
            nodeBCintersect = boundaryNodes & set(nodes)
            
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
        self.mat.assembleAll()
        self.mat.setIndices2One(indices2one)
        if not self.comm.rank:
            self.logger.info(f"KLE Matrices builded")