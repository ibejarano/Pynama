import importlib
import numpy as np

from petsc4py import PETSc

class Boundary:
    needsCoords = False
    def __init__(self, name, typ, dim):
        # TODO : Handle if func is passed as arg
        self.__name = name
        self.__type = typ
        self.__dofsConstrained = dim

    def setType(self, t):
        self.__type = t

    def getType(self):
        return self.__type

    def setValues(self, attrName, vals):
        try:
            setattr(self, attrName, np.array(vals))
        except:
            raise Exception(f"Error setting {attrName} with value; {vals} to {self.__name} boundary")

    def getValues(self, attrName, t=None, nu=None):
        try:
           val = getattr(self, attrName)
           nodesNum = len(self.getNodes())
           assert nodesNum > 0, f"Nodes not defined in boundary {self.__name} "
           arr = np.tile(val, nodesNum)
           return arr
        except AttributeError:
            raise Exception(f"{attrName} Not defined")
        return arr

    def getName(self):
        return self.__name

    def getVelocitySetted(self):
        val = getattr(self, 'velocity')
        return val

    def getDirectionsConstrained(self):
        xyz = np.array(('x', 'y', 'z'))
        return str(xyz[self.__dofsConstrained])

    def __repr__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type}  :: DOFS Constrained {self.__dofsConstrained}\n"
    
    def __str__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type}  :: DOFS Constrained {self.__dofsConstrained}\n"

    def setNodes(self, nodes: list):
        """Set Nodes that belongs to this boundary. This method transform it in a PETSc IS object that can handle dofs or nodes.

        Args:
            nodes (list): List of Nodes of this boundary
        """
        pInds = PETSc.IS().createBlock(self.__dofsConstrained, nodes)
        self.__inds = pInds
        self.__size = len(pInds.getIndices())

    def getDofsConstrained(self):
        """Returns an array with dofs constrained in this boundary
        Returns:
            [numpy array]: dofs constrained in this boundary
        """
        return self.__inds.getIndices()

    def getNodes(self):
        return self.__inds.getBlockIndices()

    def getSize(self):
        return self.__size

    def getIS(self):
        return self.__inds

    def destroy(self):
        """Free memory from the nodes or dofs saved
        """
        try:
            self.__inds.destroy()
            return self.__inds
        except:
            print("IS doesnt exists")

class NoSlipBoundary(Boundary):
    pass

class FunctionBoundary(Boundary):
    needsCoords = True
    def __init__(self, name, func_name ,attrs , dim):
        super().__init__(name, 'free-slip', dim)
        self.funcName = func_name
        self.setFunctions(attrs)

    def setFunctions(self, attrs):
        relativePath = f".{self.funcName}"
        functionLib = importlib.import_module(relativePath, package='functions')
        for attr in attrs:
            func = getattr(functionLib, attr)
            setattr(self, f"{attr}Func", func)

    def setNodesCoordinates(self, arr):
        iSet = self.getIS()
        dim = iSet.getBlockSize()
        arr.shape = ( len(self.getNodes()) , dim)
        self.__coords = arr

    def getValues(self, attrName, t, nu):
        func = getattr(self, f"{attrName}Func")
        alphaFunc = getattr(self, "alphaFunc")
        alpha = alphaFunc(t, nu)
        arr = func(self.__coords, alpha)
        return arr

    def getNodesCoordinates(self):
        return self.__coords