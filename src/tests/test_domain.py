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


class TestDomainInterfaceBoundaryConditions(unittest.TestCase):
    dataBoxMesh = {"ngl":3, "box-mesh": {
    "nelem": [2,2],
    "lower": [0,0],
    "upper": [1,1]
    }}

    dim = 2
    dim_w = 1
    sides = ['left', 'right', 'up', 'down']
    uniformValues = {"velocity": [1,5] ,"vorticity": [8] }
    bcUniform = {"uniform": uniformValues }
    bcCustomFunc = {"custom-func": {"name": "taylor_green", "attributes": ['velocity', 'vorticity', 'alpha']}}

    def create_dom(self, bc, **kwargs):
        dom = Domain()
        data = dict()
        data['domain'] = self.dataBoxMesh
        data['boundary-conditions'] = bc
        dom.configure(data)
        dom.setOptions(**kwargs)
        dom.create()
        dom.setUpIndexing()
        dom.setUpSpectralElement(Spectral(self.dataBoxMesh['ngl'], self.dim))
        dom.setUpLabels()
        dom.computeFullCoordinates()
        return dom

    def test_setup_bc_uniform(self):
        dom = self.create_dom(self.bcUniform)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

    def test_setup_bc_custom_func(self):
        if self.dim == 3:
            raise Exception("Not implemented for 3d")
        else:
            dom = self.create_dom(self.bcCustomFunc)
            dom.setUpBoundaryConditions()
            dom.setUpBoundaryCoordinates()

    def test_setup_coords_bc_custom_func(self):
        dom = self.create_dom(self.bcCustomFunc)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

    def test_create_fs_bc_uniform_custom_fc(self):
        bcData = {"free-slip": {}}
        for s in self.sides:
            bcData['free-slip'][s] = self.uniformValues
        dom = self.create_dom(bcData)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

    def test_get_fs_indices(self):
        dom = self.create_dom(self.bcUniform)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()
        dirInds = dom.getGlobalIndicesDirichlet()
        nsInds = dom.getGlobalIndicesNoSlip()

        assert len(dirInds) == 32
        assert len(nsInds) == 0

    def test_get_ns_indices(self):
        bcData = {"no-slip": {}}
        for s in self.sides:
            bcData['no-slip'][s] = self.uniformValues

        dom = self.create_dom(bcData)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()
        dirInds = dom.getGlobalIndicesDirichlet()
        nsInds = dom.getGlobalIndicesNoSlip()

        assert len(dirInds) == 0
        assert len(nsInds) == 32

    def test_set_vec_bc_constant_values_velocity(self):
        dom = self.create_dom(self.bcUniform)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

        total_nodes = len(dom.getAllNodes())
        dirInds = dom.getGlobalIndicesDirichlet()
        ref = self.uniformValues['velocity'] # in 2d [1, 5]
        arr_ref = np.zeros(total_nodes*self.dim)
        for i in list(dirInds)[::2]:
            arr_ref[i:i+self.dim] = ref
        vec_ref = PETSc.Vec().createWithArray(arr_ref)

        
        vec_test = PETSc.Vec().createSeq(total_nodes*self.dim)
        dom.applyBoundaryConditions(vec_test, "velocity")
        
        np_test.assert_array_almost_equal(vec_test, vec_ref, decimal=14)

    def test_set_vec_bc_func_all_values_velocity(self):
        dom = self.create_dom(self.bcCustomFunc)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()
        
        t = 0
        nu = 0.5/0.01

        total_nodes = len(dom.getAllNodes())
        dirInds = dom.getGlobalIndicesDirichlet()
        arr_ref = np.zeros(total_nodes*self.dim)
        for i in list(dirInds)[::2]:
            coord = dom.getNodesCoordinates([int(i/self.dim)])
            ref = velocity_test(coord[0], nu, t) 
            arr_ref[i:i+self.dim] = ref
        vec_ref = PETSc.Vec().createWithArray(arr_ref)
        
        vec_test = PETSc.Vec().createSeq(total_nodes*self.dim)
        dom.applyBoundaryConditions(vec_test, "velocity", t, nu)
        np_test.assert_array_almost_equal(vec_test, vec_ref, decimal=15)

    def test_set_vec_bc_constant_values_vorticity(self):
        dom = self.create_dom(self.bcUniform)
        dom.setUpBoundaryConditions()
        dom.setUpBoundaryCoordinates()

        total_nodes = len(dom.getAllNodes())
        dirInds = dom.getGlobalIndicesDirichlet()
        if self.dim == 2:
            dirInds = [ int(i/self.dim) for i in dirInds ]
        ref = self.uniformValues['vorticity'] # in 2d [8]
        arr_ref = np.zeros(total_nodes*self.dim_w)
        for i in dirInds:
            arr_ref[i:i+self.dim_w] = ref

        vec_ref = PETSc.Vec().createWithArray(arr_ref)

        vec_test = PETSc.Vec().createSeq(total_nodes*self.dim_w)
        dom.applyBoundaryConditions(vec_test, "vorticity")
        
        np_test.assert_array_almost_equal(vec_test, vec_ref, decimal=14)


class TestDomainMethodsBoxMesh(unittest.TestCase):
    data = {"ngl":3, "box-mesh": {
    "nelem": [5,5],
    "lower": [-5,-5],
    "upper": [8,8]
}}
    
    uniformValues = {"velocity": [1,5] ,"vorticity": [8] }
    bcUniform = {"uniform": uniformValues }

    data = {"domain": data, "boundary-conditions": bcUniform}
    
    def setUp(self):
        dom = Domain()
        dom.configure(self.data)
        dom.setOptions()
        dom.setUp()
        self.dom = dom

    def test_nodes_separation(self):
        boxData = self.data['domain']['box-mesh']
        nelem = boxData['nelem'][0]
        totalWidth = boxData['upper'][0] - boxData['lower'][0]

        width_ref = totalWidth / nelem
        width_ref /= (self.data['domain']['ngl']-1)

        width_test = self.dom.getNodeSeparationIBM() 

        assert width_ref == width_test