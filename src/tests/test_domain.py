import unittest
from domain.dmplex import BoxDom, GmshDom
from domain.domain import Domain
from domain.elements.spectral import Spectral
import numpy as np
import numpy.testing as np_test
from math import sqrt
from petsc4py import PETSc

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


class TestDomainInterfaceBoundaryConditions(unittest.TestCase):
    dataBoxMesh = {"ngl":3, "box-mesh": {
    "nelem": [2,2],
    "lower": [0,0],
    "upper": [1,1]
    }}

    dim = 2

    bcUniform = {"uniform": {"velocity": [1,0] ,"vorticity": [0] }}
    bcCustomFunc = {"custom-func": {"name": "taylor_green", "attributes": ['velocity', 'vorticity']}}

    def create_dom(self, bc, **kwargs):
        dom = Domain()
        data = dict()
        data['domain'] = self.dataBoxMesh
        data['boundary-conditions'] = bc
        dom.configure(data)
        dom.setOptions(**kwargs)
        dom.create()
        dom.setUpIndexing()
        # dim = self.__dm.getDimension()
        dom.setUpSpectralElement(Spectral(self.dataBoxMesh['ngl'], self.dim))
        dom.setUpLabels()
        dom.computeFullCoordinates()
        return dom

    def test_setup_bc_uniform(self):
        dom = self.create_dom(self.bcUniform)
        dom.setUpBoundaryConditions()

    def test_setup_bc_custom_func(self):
        dom = self.create_dom(self.bcCustomFunc)
        dom.setUpBoundaryConditions()

    def test_setup_coords_bc_custom_func(self):
        dom = self.create_dom(self.bcCustomFunc)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

    def test_create_bc_uniform_custom_fc(self):
        pass

    def test_get_fs_indices(self):
        pass

    def test_get_ns_indices(self):
        pass

    def test_set_vec_bc_constant_values(self):
        pass

    def test_set_vec_bc_func_all_values(self):
        pass

    def test_set_vec_bc_func_and_uniform_values(self):
        pass