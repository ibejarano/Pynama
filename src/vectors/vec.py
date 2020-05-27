from petsc4py import PETSc

class Vector(PETSc.Vec):
    def __init__(self):
        pass

    def create(self):
        print("new instance")

class Domain(PETSc.DMPlex):
    def __init__(self):
        self.createBoxMesh(faces=[2,2], simplex=False)
        self.view()


# vec = Vector()
# vec.create()

dom = Domain()