from utils.yaml_handler import readYaml
from domain.dmplex import BoxDM
from domain.elements.spectral import Spectral

import unittest
import numpy as np
import numpy.testing as np_test

from domain.boundaries.boundary_conditions import BoundaryConditions

class BaseBoundaryTest(unittest.TestCase):
    bcNames = ['up','down', 'right', 'left']
    valsFS = {"velocity": [1.2, 3.4], "vorticity": [0]}
    testData = {"free-slip": {
            "down": valsFS,
            "right": valsFS,
            "left": valsFS,
            "up": valsFS}}

    def setUp(self):
        config = readYaml('src/tests/dm_1')
        dm = BoxDM()
        ngl = config['domain']['ngl']
        boxMeshDict = config['domain']['box-mesh']
        nelemFaces = boxMeshDict['nelem']
        totNodes_ref = 0
        for nelemFace in nelemFaces:
            totNodes_ref+= nelemFace*2
            totNodes_ref+= (ngl-2)*nelemFace*2

        self.tot_nodes = totNodes_ref
        dm.create(boxMeshDict)
        dm.setFemIndexing(ngl)
        dim = dm.getDimension()
        spElem = Spectral(ngl, dim)
        coords = dm.computeFullCoordinates(spElem)
        
        self.dm = dm
        self.bcs = BoundaryConditions(self.dm)
        self.bcs.setBoundaryConditions(self.testData)
        self.bcs.setUp(coords)
        self.vel, _ = self.dm.createVecs()

    def test_fs_nodes_quantity(self):
        fs_test = self.bcs.getFreeSlipIndices() 
        dofs = self.dm.getDimension()
        assert len(fs_test) == self.tot_nodes * dofs

    def test_boundary_up(self):
        bc_indices = set(self.bcs.getIndicesByName("up"))
        fs_indices = self.bcs.getFreeSlipIndices()
        assert len(bc_indices - fs_indices) == 0

    def test_boundary_down(self):
        bc_indices = set(self.bcs.getIndicesByName("down"))
        fs_indices = self.bcs.getFreeSlipIndices()
        assert len(bc_indices - fs_indices) == 0

    def test_boundary_right(self):
        bc_indices = set(self.bcs.getIndicesByName("right"))
        fs_indices = self.bcs.getFreeSlipIndices()
        assert len(bc_indices - fs_indices) == 0

    def test_boundary_left(self):
        bc_indices = set(self.bcs.getIndicesByName("left"))
        fs_indices = self.bcs.getFreeSlipIndices()
        assert len(bc_indices - fs_indices) == 0

    def test_boundary_corner_upright(self):
        bc_indices_a = set(self.bcs.getIndicesByName("up"))
        bc_indices_b = set(self.bcs.getIndicesByName("right"))

        assert len(bc_indices_a.intersection(bc_indices_b)) == 2
        assert len(bc_indices_a - bc_indices_b ) == 28
        assert len(bc_indices_b - bc_indices_a ) == 28

    def test_boundary_corner_upleft(self):
        bc_indices_a = set(self.bcs.getIndicesByName("up"))
        bc_indices_b = set(self.bcs.getIndicesByName("left"))

        assert len(bc_indices_a.intersection(bc_indices_b)) == 2
        assert len(bc_indices_a - bc_indices_b ) == 28
        assert len(bc_indices_b - bc_indices_a ) == 28

    def test_boundary_corner_downright(self):
        bc_indices_a = set(self.bcs.getIndicesByName("down"))
        bc_indices_b = set(self.bcs.getIndicesByName("right"))

        assert len(bc_indices_a.intersection(bc_indices_b)) == 2
        assert len(bc_indices_a - bc_indices_b ) == 28
        assert len(bc_indices_b - bc_indices_a ) == 28

    def test_boundary_corner_downleft(self):
        bc_indices_a = set(self.bcs.getIndicesByName("down"))
        bc_indices_b = set(self.bcs.getIndicesByName("left"))

        assert len(bc_indices_a) == 30
        assert len(bc_indices_b) == 30
        assert len(bc_indices_a.intersection(bc_indices_b)) == 2
        assert len(bc_indices_a - bc_indices_b ) == 28
        assert len(bc_indices_b - bc_indices_a ) == 28

    def test_boundary_no_shared_rightleft(self):
        bc_indices_a = set(self.bcs.getIndicesByName("right"))
        bc_indices_b = set(self.bcs.getIndicesByName("left"))

        assert len(bc_indices_a.intersection(bc_indices_b)) == 0
        assert len(bc_indices_a - bc_indices_b ) == 30
        assert len(bc_indices_b - bc_indices_a ) == 30

    def test_boundary_no_shared_updown(self):
        bc_indices_a = set(self.bcs.getIndicesByName("up"))
        bc_indices_b = set(self.bcs.getIndicesByName("down"))

        assert len(bc_indices_a.intersection(bc_indices_b)) == 0
        assert len(bc_indices_a - bc_indices_b ) == 30
        assert len(bc_indices_b - bc_indices_a ) == 30
