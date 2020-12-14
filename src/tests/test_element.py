import unittest
import numpy as np
import math
import numpy.testing as np_test
from domain.elements.utilities import gaussPoints, lobattoPoints, GaussPoint2D, generateGaussPoints3D ,GaussPoint3D ,generateGaussPoints2D
from domain.elements.elemutils import SpElem2D, SpElem3D
from domain.elements.spectral import Spectral

class SpectralTest(unittest.TestCase):
    def setUp(self):
        self.spElem_ref = SpElem2D(3)
        self.spElem_test = Spectral(3, 2)

    def test_H(self):
        H_size = len(self.spElem_ref.H)
        for gps_ind in range(H_size):
            with self.subTest(test_num=gps_ind):
                H_ref = np.squeeze(np.asarray(self.spElem_ref.H[gps_ind]))
                H_test = self.spElem_test.H[gps_ind]
                np_test.assert_array_almost_equal(H_ref, H_test, decimal=12)


    def test_Hrs(self):
        H_size = len(self.spElem_ref.Hrs)
        for gps_ind in range(H_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.Hrs[gps_ind]
                test = self.spElem_test.Hrs[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)
    
    def test_HRed(self):
        H_size = len(self.spElem_ref.HRed)
        for gps_ind in range(H_size):
            with self.subTest(test_num=gps_ind):
                ref = np.squeeze(np.asarray(self.spElem_ref.HRed[gps_ind]))
                test = self.spElem_test.HRed[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HrsRed(self):
        test_size = len(self.spElem_ref.HRed)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.HrsRed[gps_ind]
                test = self.spElem_test.HrsRed[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HOp(self):
        test_size = len(self.spElem_ref.HOp)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = np.squeeze(np.asarray(self.spElem_ref.HOp[gps_ind]))
                test = self.spElem_test.HOp[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HrsOp(self):
        test_size = len(self.spElem_ref.HrsOp)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.HrsOp[gps_ind]
                test = self.spElem_test.HrsOp[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HCoo(self):
        test_size = len(self.spElem_ref.HCoo)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = np.squeeze(np.asarray(self.spElem_ref.HCoo[gps_ind]))
                test = self.spElem_test.HCoo[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HrsCoo(self):
        test_size = len(self.spElem_ref.HrsCoo)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.HrsCoo[gps_ind]
                test = self.spElem_test.HrsCoo[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)
    
    def test_HCooRed(self):
        test_size = len(self.spElem_ref.HCooRed)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = np.squeeze(np.asarray(self.spElem_ref.HCooRed[gps_ind]))
                test = self.spElem_test.HCooRed[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HrsCooRed(self):
        test_size = len(self.spElem_ref.HrsCooRed)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.HrsCooRed[gps_ind]
                test = self.spElem_test.HrsCooRed[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HCooOp(self):
        test_size = len(self.spElem_ref.HCooOp)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = np.squeeze(np.asarray(self.spElem_ref.HCooOp[gps_ind]))
                test = self.spElem_test.HCooOp[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_HrsCooOp(self):
        test_size = len(self.spElem_ref.HrsCooOp)
        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                ref = self.spElem_ref.HrsCooOp[gps_ind]
                test = self.spElem_test.HrsCooOp[gps_ind]
                np_test.assert_array_almost_equal(ref, test, decimal=12)

    def test_gps(self):
        test_size = len(self.spElem_ref.gps)

        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                gp_ref = self.spElem_ref.gps[gps_ind]
                gp_test = self.spElem_test.gps[gps_ind]

                self.assertEqual(gp_ref.r, gp_test.r)
                self.assertEqual(gp_ref.s, gp_test.s)
                self.assertEqual(gp_ref.w, gp_test.w)

    def test_gpsRed(self):
        test_size = len(self.spElem_ref.gpsRed)

        for gps_ind in range(test_size):
            with self.subTest(test_num=gps_ind):
                gp_ref = self.spElem_ref.gpsRed[gps_ind]
                gp_test = self.spElem_test.gpsRed[gps_ind]

                self.assertEqual(gp_ref.r, gp_test.r)
                self.assertEqual(gp_ref.s, gp_test.s)
                self.assertEqual(gp_ref.w, gp_test.w)

class UtilitiesElementTest(unittest.TestCase):
    def test_generate_list_gausspoints2D(self):
        for i_test in range(2,5):
            with self.subTest(i_test=i_test):
                # my new gausspoint list generator func
                gps_1D, gps_wei = gaussPoints(i_test)
                gps_list = generateGaussPoints2D(gps_1D, gps_wei)
                legacy_list = list()
                
                # this implementation was made by Alejandro
                for i in range(len(gps_1D)):
                    for j in range(len(gps_1D)):
                        legacy_list.append(GaussPoint2D(gps_1D[i], gps_1D[j], gps_wei[i]*gps_wei[j]))
                
                for i in range(len(gps_1D)**2):
                    with self.subTest(i=i):
                        gps_test = gps_list[i]
                        gps_exact = legacy_list[i]
                        np_test.assert_array_almost_equal(gps_exact.r , gps_test.r)
                        np_test.assert_array_almost_equal(gps_exact.s , gps_test.s)
                        np_test.assert_array_almost_equal(gps_exact.w , gps_test.w)


    def test_generate_list_gausspoints3D(self):
        for i_test in range(2,5):
            with self.subTest(i_test=i_test):
                # my new gausspoint list generator func
                gps_1D, gps_wei = gaussPoints(i_test)
                gps_list = generateGaussPoints3D(gps_1D, gps_wei)
                # this implementation was made by Alejandro
                legacy_list = list()
                for i in range(len(gps_1D)):
                    for j in range(len(gps_1D)):
                        for k in range(len(gps_1D)):
                            legacy_list.append(GaussPoint3D(gps_1D[i], gps_1D[j],
                            gps_1D[k], gps_wei[i]*gps_wei[j]*gps_wei[k]))
                
                for i in range(len(gps_1D)**3):
                    with self.subTest(i=i):
                        gps_test = gps_list[i]
                        gps_exact = legacy_list[i]
                        np_test.assert_array_almost_equal(gps_exact.r , gps_test.r)
                        np_test.assert_array_almost_equal(gps_exact.s , gps_test.s)
                        np_test.assert_array_almost_equal(gps_exact.t , gps_test.t)
                        np_test.assert_array_almost_equal(gps_exact.w , gps_test.w)

    def test_gausspoints_N2(self):
        # Seteo de valores exactos
        weights = np.array([1, 1], dtype=float)

        gp1 = 1 / math.sqrt(3)
        gps = np.array([ -gp1, gp1  ], dtype=float)

        # Generacion a partir de la funcion a testear
        gps_result, weights_result = gaussPoints(2)
        np_test.assert_array_equal(gps, gps_result)
        np_test.assert_array_equal(weights, weights_result) 

    def test_gausspoints_N3(self):
        # Seteo de valores exactos
        weights = np.array([ 5/9 , 8/9 , 5/9], dtype=float)

        gp1 = math.sqrt(3/5)
        gps = np.array([ -gp1, 0 ,gp1  ], dtype=float)

        # Generacion a partir de la funcion a testear
        gps_result, weights_result = gaussPoints(3)
        np_test.assert_array_almost_equal(gps, gps_result, decimal=12)
        np_test.assert_array_almost_equal(weights, weights_result, decimal=12) 

    def test_lobattopoints(self):
        # Get the exact lobatto points
        lob_poi_2 = np.array([-1, 1], dtype=float)
        wei_2 = np.array([1, 1], dtype=float)

        lob_poi_3 = np.array([-1, 0, 1], dtype=float)
        wei_3 = np.array([ 1/3, 4/3, 1/3], dtype=float)

        lob_poi_4 = np.array([-1, -1/math.sqrt(5), 1/math.sqrt(5),1], dtype=float)
        wei_4 = np.array([1/6,5/6,5/6,1/6])

        lobpoi = [lob_poi_2, lob_poi_3, lob_poi_4]
        weights = [wei_2, wei_3, wei_4]

        for i in range(3):
            with self.subTest(i=i):
                lobpoi_calculated, wei_calulated = lobattoPoints(i+2)
                np_test.assert_array_almost_equal(
                weights[i]
                ,wei_calulated
                ,decimal=12)
                np_test.assert_array_almost_equal(
                lobpoi[i]
                ,lobpoi_calculated 
                ,decimal=12)

class SpectralTestNodes(unittest.TestCase):
    def setUp(self):
        self.spectral_elements_2d = list()
        self.spectral_elements_3d = list()
        for ngl in range(2,5):
            self.spectral_elements_2d.append(Spectral(ngl,2))
            self.spectral_elements_3d.append(Spectral(ngl,3))

    def test_nnodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( (i+2) ** 2 , spectral.nnode)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual((i+2) ** 3 , spectral.nnode)

    def test_edges_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i , spectral.nnodedge)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i , spectral.nnodedge)

    def test_faces_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**2 , spectral.nnodface)

    def test_body_nodes(self):
        for i, spectral in enumerate(self.spectral_elements_2d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**2 , spectral.nnodcell)
        for i, spectral in enumerate(self.spectral_elements_3d):
            with self.subTest(test_ngl=spectral.ngl):
                self.assertEqual( i**3 , spectral.nnodcell)

class SpectralKLETest(unittest.TestCase):
    def setUp(self):
        coords_3d = [ 1,1,1 ,0,1,1, 0,0,1, 1,0,1, 1,1,0 ,1,0,0, 0,0,0, 0,1,0 ]
        ngl = 2
        self.spElem_test = Spectral(ngl, 2)
        self.spElem_test_3d = Spectral(ngl, 3)
        self.spElem_ref = SpElem2D(ngl)
        self.spElem_ref3D = SpElem3D(ngl)
        self.K_ale_2d , self.Rw_ale_2d, self.Rd_ale_2d = self.spElem_ref.getElemKLEMatrices(np.array([1,1,0,1,0,0,1,0], dtype=float))
        self.K_ale_3d , self.Rw_ale_3d, self.Rd_ale_3d = self.spElem_ref3D.getElemKLEMatrices(np.array(coords_3d, dtype=float))
        self.K , self.Rw, self.Rd = self.spElem_test.getElemKLEMatrices(np.array([1,1,0,1,0,0,1,0], dtype=float))
        self.K_3d , self.Rw_3d, self.Rd_3d = self.spElem_test_3d.getElemKLEMatrices(np.array(coords_3d, dtype=float))

        self.operators_ref = self.spElem_ref.getElemKLEOperators(np.array([1,1,0,1,0,0,1,0], dtype=float))
        self.operators_test = self.spElem_test.getElemKLEOperators(np.array([1,1,0,1,0,0,1,0], dtype=float))


    def test_K(self):
        np_test.assert_array_almost_equal(self.K_ale_2d, self.K , decimal=14)
        np_test.assert_array_almost_equal(self.K_ale_3d, self.K_3d , decimal=14)

    # TODO : Wrong Ref
    # def test_Rw(self):
    #     np_test.assert_array_almost_equal(self.Rw_ale_2d, self.Rw , decimal=10)
    #     np_test.assert_array_almost_equal(self.Rw_ale_3d, self.Rw_3d , decimal=10)

    def test_Rd(self):
        np_test.assert_array_almost_equal(self.Rd_ale_2d, self.Rd , decimal=14)
        np_test.assert_array_almost_equal(self.Rd_ale_3d, self.Rd_3d , decimal=14)


    def test_SrtElem(self):
        ref = self.operators_ref[0]
        test = self.operators_test[0]

        np_test.assert_array_almost_equal(ref, test, decimal=14)

    def test_DivElem(self):
        ref = self.operators_ref[1]
        test = self.operators_test[1]

        np_test.assert_array_almost_equal(ref, test, decimal=14)

    def test_CurlElem(self):
        ref = self.operators_ref[2]
        test = self.operators_test[2]

        np_test.assert_array_almost_equal(ref, test, decimal=14)

    def test_WeigElem(self):
        ref = np.squeeze(np.asarray(self.operators_ref[3]))
        test = self.operators_test[3]

        np_test.assert_array_almost_equal(ref, test, decimal=14)