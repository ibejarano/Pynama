from petsc4py.PETSc import IS, Vec
import numpy as np
import importlib

class BoundaryConditions:
    types = ["only FS", "only NS", "FS NS"]

    def __init__(self, sideNames):
        self.__boundaries = list()
        self.__nsBoundaries = list()
        self.__fsBoundaries = list()
        self.__type = None
        self.__ByName = dict()
        self.__ByType = { "free-slip": [], "no-slip": []}
        self.__borderNames = sideNames
        self.__dim = 2 if len(sideNames) == 4 else 3

    def __repr__(self):
        txt = " --== Boundary Conditions ==--\n"
        txt += "   Name   |   Type   |   Values   |   Dofs Contrained   \n"
        for b in self.__boundaries:
            name = b.getName()
            typ = b.getType()
            val = b.getValues()
            dirs = b.getDirectionsConstrained()
            msg = f"{name:10}|{typ:10}|{str(val):12}|{dirs:12}\n"
            txt+=msg
        return txt

    def setBoundaryConditions(self, data):
        # data its the dictionary with key 'boundary-conditions'
        assert len(data.keys()) < 3, "Wrong Boundary Conditions defined"
        if "uniform" in data:
            if "free-slip" in data or "no-slip" in data:
                print("WARNING: Only constant bc its assumed")
            self.__type = "only FS"
            # FIXME: Instead of None pass all the borders
            dim = len(vel)
            vel = data['uniform']['vel']
            vort = [0] if dim == 2 else [0, 0 , 0]
            self.__setUpBoundaries('free-slip', None, True, vel=vals, vort=vort)
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

    def getType(self):
        return self.__type

    def __setUpBoundaries(self, t, sides, cte=False, cteValues=None):
        if cte:
            for name in self.__borderNames:
                self.__setBoundary(name, t, cteValues) 
        else:   
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
        self.__ByType[typ].append(boundary)
        self.__ByName[name] = boundary

    def getNames(self, bcs=None):
        if bcs == None:
            bcs = self.__boundaries
        bNames = list()
        for b in bcs:
            bNames.append(b.getName()) 
        return bNames

    def getNamesByType(self, bcType):
        bcs = self.__ByType[bcType]
        return self.getNames(bcs)

    def setBoundaryNodes(self, bName, nodes):
        try:
            boundary = self.__ByName[bName]
            boundary.setNodes(nodes)
        except:
            raise Exception("Boundary Not found")

    def getIndicesByType(self, bcType):
        inds = IS().createGeneral([])
        boundaries = self.__ByType[bcType]
        if len(boundaries) == 0:
            return set()
        else:
            for bc in self.__ByType[bcType]:
                bcIS = bc.getIS()
                inds = bcIS.union(inds)
            return set(inds.getIndices())

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
    def __init__(self, name, typ, vel, vort, dim):
        # TODO : Handle if func is passed as arg
        self.__name = name
        self.__type = typ

        self.__vel = np.array(vel)
        self.__vort = np.array(vort)

        self.__dofsConstrained = dim

    def setType(self, t):
        self.__type = t

    def getType(self):
        return self.__type

    def setValues(self, vals):
        self.__values = np.array(vals)

    def getValues(self, node=None, t=None):
        arr = self.__values[ self.__values != None ]
        return arr

    def getName(self):
        return self.__name

    def getDirectionsConstrained(self):
        xyz = np.array(('x', 'y', 'z'))
        return str(xyz[self.__dofsConstrained])

    def __repr__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type} :: Velocity: {self.__vel} :: DOFS Constrained {self.__dofsConstrained}\n"
    
    def __str__(self):
        return f"Boundary Name:{self.__name}:: Type: {self.__type} :: Velocity: {self.__vel} :: DOFS Constrained {self.__dofsConstrained}\n"

    def setNodes(self, nodes: list):
        """Set Nodes that belongs to this boundary. This method transform it in a PETSc IS object that can handle dofs or nodes.

        Args:
            nodes (list): List of Nodes of this boundary
        """
        pInds = IS().createBlock(self.__dofsConstrained, nodes)
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


class FunctionBoundary(Boundary):

    def __init__(self, name, func_name ,attrs , dim):
        super().__init__(name, 'free-slip', None, None, dim)
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