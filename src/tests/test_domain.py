import unittest
from domain.dmplex import BoxDom, GmshDom
from domain.domain import Domain
from domain.elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

from functions.taylor_green import velocity_test, vorticity_test

class TestDomainInterface(unittest.TestCase):
    dataBoxMesh = {"ngl":3, "box-mesh": {
    "nelem": [2,2],
    "lower": [0,0],
    "upper": [1,1]
}}
    dataBoxMesh = {"domain": dataBoxMesh}
    dataGmsh = {"ngl": 3 , "gmsh-file": "src/tests/test.msh"}
    dataGmsh = {"domain": dataGmsh}

    def create_dom(self, data, **kwargs):
        dom = Domain()
        dom.configure(data)
        dom.setOptions(**kwargs)
        dom.create()
        dom.setUpIndexing()
        return dom

    def test_create_boxmesh(self):
        dom = self.create_dom(self.dataBoxMesh)
        test_type = dom.getMeshType()
        test_ngl = dom.getNGL()
        test_numOfElem = dom.getNumOfElements()
        test_numOfNodes = dom.getNumOfNodes()
        assert test_type == 'box'
        assert test_numOfElem == 4
        assert test_ngl == 3
        assert test_numOfNodes == 25

    def test_create_gmsh(self):
        dom = self.create_dom(self.dataGmsh)
        test_type = dom.getMeshType()
        test_ngl = dom.getNGL()
        test_numOfElem = dom.getNumOfElements()
        test_numOfNodes = dom.getNumOfNodes()
        assert test_type == 'gmsh'
        assert test_ngl == 3
        assert test_numOfElem == 33
        assert test_numOfNodes == 153 # This number is from Gmsh

    def test_box_set_from_opts_ngl(self):
        ngl_ref = 7
        dom = self.create_dom(self.dataBoxMesh, ngl=ngl_ref)
        ngl_test = dom.getNGL()

        nelem = self.dataBoxMesh['domain']['box-mesh']['nelem']
        ref_numOfNodes = (ngl_ref*nelem[0] - 1)*(ngl_ref*nelem[1] - 1)

        test_numOfNodes = dom.getNumOfNodes()
        assert ngl_test == ngl_ref
        assert test_numOfNodes == ref_numOfNodes

    def test_gmsh_set_from_opts_ngl(self):
        ngl_ref = 8
        dom = self.create_dom(self.dataGmsh, ngl=ngl_ref)
        test_numOfNodes = dom.getNumOfNodes()

        ngl_test = dom.getNGL()
        assert ngl_test == ngl_ref
        assert test_numOfNodes == 1688 # This number is from Gmsh

    def test_set_from_opts_nelem(self):
        dom = self.create_dom(self.dataBoxMesh, nelem=[4,4])
        test_numOfElem = dom.getNumOfElements()
        assert test_numOfElem == 16

    def test_set_from_opts_hmin(self):
        pass