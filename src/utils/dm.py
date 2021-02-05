import numpy as np

def getTotalCells(dm):
    firstCell, lastCell = dm.getHeightStratum(0)
    return lastCell - firstCell

def getCellRange(self):
    return self.getHeightStratum(0)

def getCellCornersCoords(dm, startCell ,cell):
    coordinates = dm.getCoordinatesLocal()
    coordSection = dm.getCoordinateSection()
    return dm.vecGetClosure(coordSection, coordinates, cell+startCell)

def getEdgesWidth(dm):
    startEnt, _ = dm.getDepthStratum(1)
    coordinates = dm.getCoordinatesLocal()
    coordSection = dm.getCoordinateSection()
    coord = dm.vecGetClosure(coordSection, coordinates, startEnt).reshape(2,self.dim)
    coord = coord[1] - coord[0]
    norm = np.linalg.norm(coord)
    return norm

def getDofsRange(dm):
    sec = dm.getGlobalSection()
    return sec.getOffsetRange()