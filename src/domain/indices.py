# import PETSc from petsc4py
import numpy as np
import logging

class IndicesManager:
    def __init__(self, dim, ngl, comm):
        # self.indices = PETSc.IS()
        self.logger = logging.getLogger("[{}] IndicesManager Class".format(comm.rank) )
        self.comm = comm
        self.dim = dim
        self.reorderEntities = self.reorderEntities2D if dim ==2 else self.reorderEntities3D
        self._ngl = ngl
        self.__dirNodes = set()
        self.__nsNodes = set()
         
        if not comm.rank:
            self.logger.debug("IndicesManager Seteado")

    def getNGL(self):
        return self._ngl

    def getNumCompAndNumDof(self, componentsPerField ,numFields):
        numComp = [componentsPerField] * numFields
        nodesPerEntity = [ 1 , self._ngl - 2 , (self._ngl - 2)**2 ]
        #           vertex-^ edges -^     faces -^   
        if self.dim == 3:
            nodesPerEntity.append((self._ngl - 2)**3)
            # append body nodes     -^
        numDof = [componentsPerField * nodes for nodes in nodesPerEntity]
        return numComp, numDof

    def setLocalIndicesSection(self, indicesSection):
        self._indSection = indicesSection
        if not self.comm.rank:
            self.logger.debug("Section Indices Locales setUp")

    def getLocalIndicesSection(self):
        return self._indSection

    def setGlobalIndicesSection(self, indicesSection):
        self._globalIndicesSection = indicesSection
        if not self.comm.rank:
            self.logger.debug("Section Indices Globales setUp")

    def getGlobalIndicesSection(self):
        return self._globalIndicesSection

    def getTotalNodes(self):
        total = self._globalIndicesSection.getStorageSize()
        return total

    def setDirichletNodes(self, nodes: set):
        self.__dirNodes |= nodes

    def getDirichletNodes(self):
        self.globalIndicesDIR = set()
        for remoteIndices in self.comm.allgather(self.__dirNodes):
            self.globalIndicesDIR |= remoteIndices
        return self.__dirNodes

    def setNoSlipNodes(self, nodes: set):
        self.__nsNodes |= nodes

    def getNoSlipNodes(self):
        self.globalIndicesNS = set()
        for remoteIndices in self.comm.allgather(self.__nsNodes):
            self.globalIndicesNS |= remoteIndices
        return self.__nsNodes

    def mapEntitiesToNodes(self, entities, orientations, getShared = True):
        entitites = self.reorderEntities(entities)
        orientations = self.reorderEntities(orientations)
        nodes = list()
        for enu, entity in enumerate(entitites):
            nodesInEntity, ownNode = self.getGlobalNodes(entity)
            if ownNode or getShared:
                if self.dim == 2:
                    if orientations[enu]:
                        nodesInEntity.reverse()
                else:
                    if enu in [10,11,14,15,16,18]:
                        nodesInEntity.reverse()
                    if enu>19:
                        if orientations[enu]:
                            nodesInEntity.reverse()

                nodes += nodesInEntity
        #     indPoiList.append(indPoi)
        #     ownPoiList.append(ownPoi)
        # self.logger.debug("indice tipo %s indpoi %s" , indType , indPoiList)
        # self.logger.debug("indice tipo %s ownpoi %s", indType ,ownPoiList)
        return nodes

    def mapNodesToIndices(self, nodes, dof):
        return [x*dof + d for x in nodes
                            for d in range(dof)]

    def getLocalNodes(self, entity):
        localNodes, _ = self.getSectionOffset(self._indSection, entity)
        return localNodes

    def getGlobalNodes(self, entity, shared=True):
        globalNodes, ownNodes = self.getSectionOffset(self._globalIndicesSection, entity)
        if not shared:
            if ownNodes:
                return globalNodes, ownNodes
            else:
                return [],[]
        return globalNodes, ownNodes

    def getSectionOffset(self, section, poi):
        pOffset = section.getOffset(poi)
        pDof = section.getDof(poi)
        ownPoi = (pDof >= 0)
        pOffset = pOffset if ownPoi else -(pOffset + 1)
        pDof = pDof if ownPoi else -(pDof + 1)
        # self.logger.debug("getSectionOffset pOffset {s} pof {} ownPoi {} ".format(pOffset, pDof, ownPoi))
        return [pOffset + ii for ii in range(pDof)], ownPoi

    @staticmethod
    def reorderEntities2D(entities):
        return np.hstack((entities[5:], entities[1:5], entities[0]))

    @staticmethod
    def reorderEntities3D(entities):
        return np.hstack((entities[19:], entities[7:19], entities[1:7], entities[0]))