from petsc4py.PETSc import IS, Vec
import numpy as np
import logging
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
        self.__needsCoords = list()
        self.__borderNames = sides
        self.__dim = 2 if len(sides) == 4 else 3

        self.logger = logging.getLogger("Boundary Conditions:")

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
                self.logger.warning("WARNING: Only constant bc its assumed")
            self.__type = "FS"
            valsDict = data['uniform']
            self.__setUpBoundaries('free-slip', self.__borderNames, valsDict)
        elif "custom-func" in data:
            self.__type = "FS"
            funcName = data['custom-func']['name']
            attrs = data['custom-func']['attributes']
            self.__setFunctionBoundaries(funcName, attrs)
        elif "free-slip" in data and "no-slip" in data:
            self.__type = "FS-NS"
            self.__setPerBoundaries('free-slip', data['free-slip'])
            self.__setPerBoundaries('no-slip', data['no-slip'])
        elif "free-slip" in data:
            self.__type = "FS"
            self.__setPerBoundaries('free-slip', data['free-slip'])
        elif "no-slip" in data:
            self.__type = "NS"
            self.__setPerBoundaries('no-slip', data['no-slip'])
        else:
            raise Exception("Boundary Conditions not defined")

    def getType(self):
        return self.__type

    def __setUpBoundaries(self, t, sides, vals: dict):
        for nameSide in sides:
            self.__setBoundary(nameSide, t, vals)

    def __setPerBoundaries(self, t, sidesDict: dict):
        for name, vals in sidesDict.items():
            if "custom-func" in vals:
                funcName = vals['custom-func']['name']
                attrs = vals['custom-func']['attributes']
                self.__setFunctionBoundary(name, funcName, attrs)
            else:
                self.__setBoundary(name, t , vals)

    def __setBoundary(self, name, typ, vals: dict):
        boundary = Boundary(name, typ, self.__dim)

        for attrName, val in vals.items():
            boundary.setValues(attrName, val)

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
        self.__fsBoundaries.append(boundary)
        self.__ByName[borderName] = boundary
        self.__ByType['free-slip'].append(boundary)
        self.__needsCoords.append(borderName)

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

    def getBordersNeedsCoords(self):
        return self.__needsCoords

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

