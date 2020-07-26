from petsc4py import PETSc
import logging
from viewer.xml_generator import XmlGenerator
import os

access_rights = 0o755

class Paraviewer:
    def __init__(self, dim, comm, saveDir=None):
        self.comm = comm
        self.saveDir = '.' if not saveDir else saveDir
        if not os.path.isdir(self.saveDir):
            os.makedirs(f"./{self.saveDir}")
        self.xmlWriter = XmlGenerator(dim)

    def saveMesh(self, coords, name='mesh'):
        totalNodes = int(coords.size / self.xmlWriter.dim)
        self.xmlWriter.setUpDomainNodes(totalNodes=totalNodes)
        self.xmlWriter.generateXMLTemplate()

        coords.setName(name)
        ViewHDF5 = PETSc.Viewer()
        try:
            ViewHDF5.createHDF5(f'{self.saveDir}/mesh.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)
        except:
            os.makedirs(f"./{self.saveDir}")
            ViewHDF5.createHDF5(f'./{self.saveDir}/mesh.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)

        ViewHDF5.view(obj=coords)
        ViewHDF5.destroy()

    def saveData(self, step, time, *vecs):
        for vec in vecs:
            self.saveVec(vec, step)
        self.saveStepInXML(step, time, vecs=vecs)

    def saveVec(self, vec, step=None):
        """Save the vector."""
        name = vec.getName()
        # self.logger.debug("saveVec %s" % name)
        ViewHDF5 = PETSc.ViewerHDF5()     # Init. Viewer

        if step is None:
            ViewHDF5.create(f"./{self.saveDir}/{name}.h5", mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)
        else:
            ViewHDF5.create(f"./{self.saveDir}/{name}-{step:05d}.h5",
                            mode=PETSc.Viewer.Mode.WRITE, comm=self.comm)
        ViewHDF5.pushGroup('/fields')
        ViewHDF5.view(obj=vec)   # Put PETSc object into the viewer
        ViewHDF5.destroy()            # Destroy Viewer

    def saveStepInXML(self, step, time, vec=None ,vecs=None):
        dataGrid = self.xmlWriter.generateMeshData("mesh1")
        self.xmlWriter.setTimeStamp(time, dataGrid)
        try:
            self.xmlWriter.setVectorAttribute(vec.getName(), step, dataGrid)
        except:
            for vec in vecs:
                if vec.getSize() == self.xmlWriter.dimensions:
                    self.xmlWriter.setScalarAttribute(vec.getName(), step, dataGrid)
                else:
                    self.xmlWriter.setVectorAttribute(vec.getName(), step, dataGrid)

    def writeXmf(self, name):
        self.xmlWriter.writeFile(f"./{self.saveDir}/{name}")