from utils.yaml_handler import readYaml
from domain.dmplex import BoxDM
from domain.elements.spectral import Spectral

import unittest
import numpy as np
import numpy.testing as np_test

from domain.boundaries.boundary_conditions import BoundaryConditions

class BaseBoundaryTest(unittest.TestCase):
    bcNames = []
    testData = dict() 

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

class TestBoundaryConditionsUniform(BaseBoundaryTest):
    bcNames = ['up','down', 'right', 'left']
    valsFS = {"velocity": [1.2, 3.4], "vorticity": [0]}
    testData = {"free-slip": {
            "down": valsFS,
            "right": valsFS,
            "left": valsFS,
            "up": valsFS}}

    def test_set_boundaries(self):
        assert "FS" == self.bcs.getType()
        bcsFSNames = self.bcs.getNamesByType('free-slip')
        bcsNSNames = self.bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

    def test_fs_nodes_quantity(self):
        fs_test = self.bcs.getFreeSlipIndices() 
        dofs = self.dm.getDimension()
        assert len(fs_test) == self.tot_nodes * dofs

    def test_set_boundary_conditions(self):
        # get the global vel vec (n-size)
        self.vel.set(0.0)
        n = self.vel.getSize()
        fs_dofs = len(self.bcs.getFreeSlipIndices())
        # set bc (use the actual function to test)
        self.bcs.setValuesToVec(self.vel, 'velocity', t=0, nu=1)
        vel_array = self.vel.getArray()
        unique, counts = np.unique(vel_array, return_counts=True)
        count_values = dict(zip(unique, counts))

        dim = self.dm.getDimension()
        bcs_x = int(fs_dofs / dim)
        bcs_y = int(fs_dofs / dim)
        non_bc = n - bcs_x - bcs_y

        assert count_values[0.0] == non_bc
        assert count_values[1.2] == bcs_x
        assert count_values[3.4] == bcs_y

class TestBoundaryConditionsCustomFunc(BaseBoundaryTest):
    bcNames = ['up','down', 'right', 'left']
    custFS = {"name": 'taylor_green', "attributes": ['velocity', 'vorticity']}
    testData = {"custom-func": custFS}

    def test_set_boundaries(self):
        assert "FS" == self.bcs.getType()
        bcsFSNames = self.bcs.getNamesByType('free-slip')
        bcsNSNames = self.bcs.getNamesByType('no-slip')
        assert len(bcsFSNames) == len(self.bcNames)
        assert bcsNSNames == []

        bordersThatNeedsCoords = self.bcs.getBordersNeedsCoords()
        for bName in self.bcNames:
            assert bName in bordersThatNeedsCoords
class TestBoundaryConditionsFSNS(BaseBoundaryTest):
    bcNames = ['up','down', 'right', 'left']
    valsFS = {"velocity": [1,0], "vorticity": [0]}
    valsNS = {"velocity": [2,0]}
    testData = {"free-slip": {
        "down": valsFS,
        "right": valsFS},
            "no-slip": {
        "left": valsNS,
        "up": valsNS}}

    def test_get_indices(self):
        nodes_down = [0 , 1 , 2 , 3]
        nodes_right = [3 , 4, 5, 6]
        nodes_up = [6 , 7 , 8]
        nodes_left = [8 , 9 , 10 , 11, 0]

        self.bcs.setBoundaryNodes("down", nodes_down)
        self.bcs.setBoundaryNodes("up", nodes_up)
        self.bcs.setBoundaryNodes("left", nodes_left)
        self.bcs.setBoundaryNodes("right", nodes_right)

        # no slip are left and up so...
        ns_nodes = nodes_left + nodes_up
        dim = 2
        ns_indices = [n*dim + dof for n in ns_nodes for dof in range(dim)]
        fs_nodes = nodes_down + nodes_right
        fs_indices = [n*dim + dof for n in fs_nodes for dof in range(dim)]

        assert set(ns_indices) == self.bcs.getNoSlipIndices()
        assert set(fs_indices) == self.bcs.getFreeSlipIndices()
        assert set(ns_indices) == self.bcs.getIndicesByType('no-slip')
        assert set(fs_indices) == self.bcs.getIndicesByType('free-slip')
