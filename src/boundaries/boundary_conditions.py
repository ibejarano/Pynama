from petsc4py.PETSc import IS, Vec
import numpy as np
from .boundary import Boundary, FunctionBoundary

class BoundaryConditions:
    types = ["FS", "NS", "FS-NS"]
    bcTypesAvailable = ("uniform", "custom-func", "free-slip", "no-slip")
    def __init__(self, sides: list):
        self.__boundaries = list()
        self.__nsBoundaries = list()
        self.__fsBoundaries = list()
        self.__type = None
        self.__ByName = dict()
        self.__ByType = { "free-slip": [], "no-slip": []}

        self.__borderNames = list()
        for s in sides:
            self.__borderNames = b["name"]
        self.__dim = 2 if len(sides) == 4 else 3

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
            self.__type = "FS-NS"
            self.__setUpBoundaries('free-slip', data['free-slip'])
            self.__setUpBoundaries('no-slip', data['no-slip'])
        elif "free-slip" in data:
            self.__type = "FS"
            self.__setUpBoundaries('free-slip', data['free-slip'])
        elif "no-slip" in data:
            self.__type = "NS"
            self.__setUpBoundaries('no-slip', data['no-slip'])
        else:
            raise Exception("Boundary Conditions not defined")

    def newsetBC(self, data):
        bcTypes = data.keys()
        if 'uniform' in bcTypes:
            pass
        elif 'custom-func' in bcTypes:
            pass
        elif "free-slip" in bcTypes and "no-slip" in bcTypes:
            pass
        elif "free-slip" in bcTypes:
            pass
        elif "no-slip" in bcTypes:
            pass
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

    def __setFunctionBoundaries(self, funcName, attrs):
        for borderName in self.__borderNames:
            self.__setFunctionBoundary(borderName, funcName, attrs)

    def __setFunctionBoundary(self, borderName, funcName, attrs):
        dim = self.__dim
        boundary = FunctionBoundary(borderName , funcName , attrs , dim)
        boundary.setNodes(self.nodes)
        boundary.setNodesCoordinates(self.coords)
        self.__fsBoundaries.append(boundary)

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

