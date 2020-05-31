from petsc4py import PETSc
import logging
from viewer.xml_generator import XmlGenerator

class Paraviewer:
    def __init__(self, dim, comm):
        self.comm = comm

        #hacer en paralelo aun
        self.xmlWriter = XmlGenerator(dim)

    def saveMesh(self, coords, name='mesh'):
        totalNodes = int(coords.size / self.xmlWriter.dim)
        self.xmlWriter.setUpDomainNodes(totalNodes=totalNodes)
        self.xmlWriter.generateXMLTemplate()

        coords.setName(name) #coords must be a PETSc vec obj
        ViewHDF5 = PETSc.Viewer()     # Init. Viewer
        ViewHDF5.createHDF5(name + '.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)

        # self.fullCoordVec.view()
        ViewHDF5.view(obj=coords)  # Put PETSc object into viewer
        ViewHDF5.destroy()

    def saveVec(self, vec, timeStep=None):
        """Save the vector."""
        name = vec.getName()
        # self.logger.debug("saveVec %s" % name)
        ViewHDF5 = PETSc.ViewerHDF5()     # Init. Viewer

        if timeStep is None:
            ViewHDF5.create(name + '.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)
        else:
            ViewHDF5.create(name + '-%04d.h5' % timeStep,
                            mode=PETSc.Viewer.Mode.WRITE, comm=self.comm)
        ViewHDF5.pushGroup('/fields')
        ViewHDF5.view(obj=vec)   # Put PETSc object into the viewer
        ViewHDF5.destroy()            # Destroy Viewer

    def saveStepInXML(self, vec, step, time):
        dataGrid = self.xmlWriter.generateMeshData("mesh1")
        self.xmlWriter.setTimeStamp(time, dataGrid)
        self.xmlWriter.setAttribute(vec.getName(), step, dataGrid)

    def writeXmf(self, name):
        self.xmlWriter.writeFile(name)