import numpy as np
from math import floor

def reorderEntities(entities):
    return np.hstack((entities[5:], entities[1:5], entities[0]))

def getLocalDofsFromCell(dm, cell):
    points, oris = dm.getTransitiveClosure(cell)
    arr = np.zeros(0, dtype=np.int32)
    points = reorderEntities(points)
    oris = dm.reorderEntities(oris)
    for i, poi in enumerate(points):
        arrtmp = np.arange(*dm.getPointLocal(poi))
        if oris[i] == -2:
            tmp = arrtmp.copy()
            tmp[-2::-2] = arrtmp[::2]
            tmp[::-2] = arrtmp[1::2]
            arrtmp = tmp
        arr = np.append(arr, arrtmp)
    return arr.astype(np.int32)

def getGlobalDofsFromCell(dm, cell):
    points, _ = dm.getTransitiveClosure(cell)
    arr = np.zeros(0, dtype=np.int32)
    points = reorderEntities(points)
    for poi in points:
        arrtmp = np.arange(*dm.getPointGlobal(poi))
        arr = np.append(arr, arrtmp)
    return arr.astype(np.int32)

def getCoordinates(dm, coordVec, indices=None, nodes=None):
    if nodes:
        raise Exception("Not implemented for nodes yet")
    dim = dm.getDimension()
    numOfNodes = floor(len(indices) / dim)
    arr = coordVec.getValues(indices).reshape((numOfNodes,dim))
    return arr