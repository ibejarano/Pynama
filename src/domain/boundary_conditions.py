from petsc4py.PETSc import IS
import numpy as np


class BoundaryConditions:
    types = ["constant", "only FS", "only NS", "FS NS"]

    def __init__(self):
        self.__boundaries = list()
        self.__nsBoundaries = list()
        self.__fsBoundaries = list()
        self.__type = None

    def setBoundaryConditions(self, data):
        # data its the dictionary with key 'boundary-conditions'
        assert len(data.keys()) < 3, "Wrong Boundary Conditions defined"
        if "constant" in data:
            if "free-slip" in data or "no-slip" in data:
                print("WARNING: Only constant bc its assumed")
            self.__type = "constant"
            # FIXME: Instead of None pass all the borders
            self.__setUpBoundaries(data['constant'], None)
        elif "free-slip" in data and "no-slip" in data:
            self.__type = "FS NS"
            self.__setUpBoundaries('free-slip', data['free-slip'])
            self.__setUpBoundaries('no-slip', data['no-slip'])
        elif "free-slip" in data:
            self.__type = "only FS"
            self.__setUpBoundaries('free-slip', data['free-slip'])
        elif "no-slip" in data:
            self.__type = "only NS"
            self.__setUpBoundaries('no-slip', data['no-slip'])
        else:
            raise Exception("Boundary Conditions not defined")
        # print(self.__boundaries)

    def __setUpBoundaries(self, t, sides):
        for name, vals in sides.items():
            self.__setBoundary(name, t, vals)

    def __setBoundary(self, name, typ, values):
        boundary = Boundary(name, typ, values)
        self.__boundaries.append(boundary)
        if typ == 'free-slip':
            self.__fsBoundaries.append(boundary)
        elif typ == 'no-slip':
            self.__nsBoundaries.append(boundary)
        else:
            raise Exception("Wrong boundary type")

    def getBoundaries(self):
        return self.__boundaries

    def getNoSlipIndices(self):
        inds = IS().createGeneral([])
        for bc in self.__nsBoundaries:
            bcIS = bc.getIS()
            inds = bcIS.union(inds)
        return set(inds.getIndices())

    def getFreeSlipIndices(self):
        inds = IS().createGeneral([])
        for bc in self.__fsBoundaries:
            bcIS = bc.getIS()
            inds = bcIS.union(inds)
        return set(inds.getIndices())

class Boundary:
    def __init__(self, name, t, values):
        self.__name = name
        self.__type = t
        self.__values = values
        self.__dofsConstrained = []  # 0, 1 ( 2 for dim = 3)
        for dof, val in enumerate(values):
            if val == None:
                pass
            else:
                self.__dofsConstrained.append(dof)

    def setType(self, t):
        self.__type = t

    def setValues(self, vals):
        self.__values = vals

    def getName(self):
        return self.__name

    def __repr__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type} :: Values: {self.__values} :: DOFS Constrained {self.__dofsConstrained} \n "

    def setIndices(self, nodes: list):
        """Set Degrees of freedom that belongs to this boundary

        Args:
            nodes (list): List of Nodes of this boundary
        """
        dofs = len(self.__values)
        pInds = IS().createBlock(dofs, nodes)
        self.__inds = pInds

    def getIndices(self):
        return self.__inds.getIndices()

    def getNodes(self):
        return self.__inds.getBlockIndices()

    def getIS(self):
        return self.__inds

    def destroy(self):
        """Free memory from the nodes or dofs saved
        """
        try:
            self.__inds.destroy()
        except:
            print("IS doesnt exists")


if __name__ == "__main__":
    testData = {
        "free-slip": {
            "down": [None, 0],
            "left": [1, 0],
            "right": [1, 0]},
        "no-slip": {
            "up": [1, 1]
        }
    }

    bcs = BoundaryConditions()
    bcs.setBoundaryConditions(testData)