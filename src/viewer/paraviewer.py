from petsc4py import PETSc
import logging
from viewer.xml_generator import XmlGenerator
import os
import yaml

access_rights = 0o755

class Paraviewer:
    comm = PETSc.COMM_WORLD
    def __init__(self):
        self.logger = logging.getLogger(f"[{self.comm.rank}]:Viewer:")

    def configure(self, dim, saveDir=None):
        self.saveDir = '.' if not saveDir else saveDir
        if not self.comm.rank and not os.path.isdir(self.saveDir):
            os.makedirs(f"./{self.saveDir}")
        self.h5name = "vec-data"
        self.xmlWriter = XmlGenerator(dim, self.h5name)

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
        self.logger.debug("Mesh saved")
        ViewHDF5.destroy()

    def saveData(self, step, time, *vecs):
        self.saveVec(vecs, step)
        self.saveStepInXML(step, time, vecs=vecs)
        self.logger.debug("Step saved")

    def saveVec(self, vecs, step):
        name = self.h5name
        # self.logger.debug("saveVec %s" % name)
        ViewHDF5 = PETSc.ViewerHDF5()     # Init. Viewer
        ViewHDF5.create(f"./{self.saveDir}/{name}-{step:05d}.h5",
                            mode=PETSc.Viewer.Mode.WRITE, comm=self.comm)
        ViewHDF5.pushGroup('/fields')
        for vec in vecs:
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

    def writeVTK(self, name, dm , step=None):
        viewer = PETSc.Viewer()
        if step == None:
            viewer.createVTK(f"./{self.saveDir}/{name}.vtk", mode=PETSc.Viewer.Mode.WRITE)
        else:
            viewer.createVTK(f"./{self.saveDir}/{name}-{step:05d}.vtk", mode=PETSc.Viewer.Mode.WRITE)
        viewer.view(obj=dm)
        viewer.destroy()       

    def writeXmf(self, name):
        self.xmlWriter.writeFile(f"./{self.saveDir}/{name}")

    def writeYaml(self, name, data):
        data['dir'] = self.saveDir
        with open(self.saveDir+'.yaml', 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False)