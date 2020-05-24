from petsc4py import PETSc
import logging

class Paraviewer:
    def __init__(self, comm):
        self.comm = comm

    def saveMesh(self, coords, name='mesh'):
        coords.setName(name) #coords must be a PETSc vec obj
        ViewHDF5 = PETSc.Viewer()     # Init. Viewer
        ViewHDF5.createHDF5(name + '.h5', mode=PETSc.Viewer.Mode.WRITE,
                            comm=self.comm)

        # self.fullCoordVec.view()
        ViewHDF5.view(obj=coords)  # Put PETSc object into viewer
        ViewHDF5.destroy()