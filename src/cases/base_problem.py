from mpi4py import MPI
import logging
# Local packages
from domain.dmplex import DMPlexDom
from elements.spectral import Spectral
from viewer.paraviewer import Paraviewer
from solver.ts_solver import TsSolver

class BaseProblem(object):
    def __init__(self, comm=MPI.COMM_WORLD):
        """
        comm: MPI Communicator
        """
        self.comm = comm
        self.logger = logging.getLogger("Setting Up Base Problem")

    def setUpDomain(self):
        self.dom = DMPlexDom(self.lower, self.upper, self.nelem)
        self.logger.debug("DMPlex dom intialized")

    def setUpHighOrderIndexing(self):
        self.dom.setFemIndexing(self.ngl)

    def setUpElement(self):
        self.elemType = Spectral(self.ngl, self.dim)

    def setUpWithInputData(self, inputData):
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

    def setUpGeneral(self, inputData):
        self.setUpWithInputData(inputData)
        self.setUpDomain()
        self.setUpHighOrderIndexing()
        self.setUpElement()
        self.createMesh()

    def getTS(self):
        ts = TsSolver(self.comm)
        return ts