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
        if not comm.rank:
            self.logger.debug("IndicesManager Seteado")

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

    def setDirichletIndices(self, bcInput):
        self.BC2nodedict = bcInput

    def getDirichletIndices(self):
        self.globalIndicesDIR = set()
        localIndicesDIR = set()
        for nodes in self.BC2nodedict.values():
            localIndicesDIR |= nodes

        for remoteIndices in self.comm.allgather(localIndicesDIR):
            self.globalIndicesDIR |= remoteIndices

        return localIndicesDIR

    def getNoSlipIndices(self, bcInput):
        indicesNS = set()
        if 'no-slip-border' in bcInput:
            noSlipBorders = bcInput['no-slip-border']
            for border in noSlipBorders:
                indices = []
                # self.logger.debug("border con ns: %s", 2**border)
                # self.logger.debug("indices encontrados (nodos?): %s", indices)

            # if self.BCdict[bc]['ns'] is not None:
            #     indicesNS |= self.BC2nodedict[bc]

        # global indices for DIR and NS BC are allgathered amondg processes
        globIndices = self.comm.tompi4py().allgather([indicesNS])
        return globIndices

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

    def getGlobalNodes(self, entity):
        globalNodes, ownNodes = self.getSectionOffset(self._globalIndicesSection, entity)
        # self.logger.debug("entity  %s", entity)
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