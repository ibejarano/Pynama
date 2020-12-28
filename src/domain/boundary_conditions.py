class BoundaryConditions:
    types = ["constant", "only FS", "only NS", "FS NS"]
    def __init__(self):
        self.__boundaries = list()
        self.__type = None

    def setBoundaryConditions(self, data):
        # data its the dictionary with key 'boundary-conditions'
        assert len(data.keys()) < 3 , "Wrong Boundary Conditions defined"
        if "constant" in data:
            if "free-slip" in data or "no-slip" in data:
                print("WARNING: Only constant bc its assumed")
            self.__type = "constant"
            self.setUpBoundaries(data['constant'])
        elif "free-slip" in data and "no-slip" in data:
            self.__type = "FS NS"
            self.setUpBoundaries('free-slip', data['free-slip'])
            self.setUpBoundaries('no-slip', data['no-slip'])
        elif "free-slip" in data:
            self.__type = "only FS"
            self.setUpBoundaries('free-slip', data['free-slip'])
        elif "no-slip" in data:
            self.__type = "only NS"
            self.setUpBoundaries('no-slip', data['no-slip'])
        else:
            raise Exception("Boundary Conditions not defined")
        print(self.__boundaries)

    def setUpBoundaries(self, t , sides):
        for name, vals in sides.items():
            self.setBoundary(name, t ,vals)

    def setBoundary(self, name, typ, values):
        boundary = Boundary(name, typ, values)
        self.__boundaries.append(boundary)

class Boundary:
    def __init__(self, name, t, values):
        self.__name = name
        self.__type = t
        self.__values = values
        self.__dofsConstrained = [] # 0, 1 ( 2 for dim = 3)
        for dof, val in enumerate(values):
            if val == None:
                pass
            else:
                self.__dofsConstrained.append(dof)

    def setType(self, t):
        self.__type = t

    def setValues(self, vals):
        self.__values = vals

    def __repr__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type} :: Values: {self.__values} :: DOFS Constrained {self.__dofsConstrained} \n "
    
    def setIndices(self, nodes=None, inds=None):
        """Set Degrees of freedom that belongs to this boundary

        Args:
            nodes (list, optional): List of Nodes of this boundary. Defaults to None.
            inds (list, optional): List of DOfs of this boudnary. Defaults to None.
        """
        raise Exception("Not implemented yet")

    def getIndices(self):
        raise Exception("Not implemented yet")

    def destroy(self):
        """Free memory from the nodes or dofs saved
        """
        raise Exception("Not implemented yet")

if __name__ == "__main__":
    testData = {
        # "constant": {
        #     "re": 100
        # }
        "free-slip": {
            "up": [ None , 0],
            "down": [ None, 0],
            "left": [1, 0],
            "right": [1, 0]},
        "no-slip": {
            "up": [1,1]
        }
    }

    bcs = BoundaryConditions()
    bcs.setBoundaryConditions(testData)
