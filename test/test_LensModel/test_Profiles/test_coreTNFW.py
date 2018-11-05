__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.coreTNFW import coreTNFW
import numpy as np
import numpy.testing as npt
import pytest


class TestcoreTNFW(object):

    def setup(self):

        self.tnfw = TNFW()
        self.tcore = coreTNFW()

    def test_deflection(self):


        Rs = 0.1
        x = np.linspace(Rs * 0.0001, Rs * 100, 10000)
        y = np.linspace(Rs * 0.001, Rs * 100, 10000)
        theta_Rs = 0.1

        tau = 10
        p = 0.7

        r_trunc = Rs * tau
        r_core = p * Rs

        xdef_t1, ydef_t1 = self.tnfw.derivatives(x, y, Rs, theta_Rs, r_trunc)
        xdef_t2, ydef_t2 = self.tnfw.derivatives(x, y, Rs, theta_Rs, r_core)
        xdefcore, ydefcore = self.tcore.derivatives(x, y, Rs, theta_Rs, r_trunc, r_core)

        np.testing.assert_almost_equal(xdef_t1 - xdef_t2, xdefcore, 5)
        np.testing.assert_almost_equal(ydef_t1 - ydef_t2, xdefcore, 5)

    def test_potential(self):
        Rs = 0.1
        x = np.linspace(Rs * 0.0001, Rs * 100, 10000)
        y = np.linspace(Rs * 0.001, Rs * 100, 10000)
        theta_Rs = 0.1

        tau = 10
        p = 0.7

        r_trunc = Rs * tau
        r_core = p * Rs

        pot = self.tnfw.nfwPot((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc)
        potcore = self.tcore.nfwPot((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc, r_core)

        np.testing.assert_almost_equal(pot, potcore, 4)

    def test_gamma(self):
        Rs = 0.1
        x = np.linspace(Rs * 0.0001, Rs * 100, 10000)
        y = np.linspace(Rs * 0.001, Rs * 100, 10000)
        theta_Rs = 0.1

        tau = 10
        p = 0.7

        r_trunc = Rs * tau
        r_core = p * Rs

        g1t, g2t = self.tnfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc, x, y)
        g1t2, g2t2 = self.tnfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_core, x, y)

        g1, g2 = self.tcore.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc, r_core, x, y)

        np.testing.assert_almost_equal(g1t-g1t2, g1, 5)
        np.testing.assert_almost_equal(g2t-g2t2, g2, 5)

    def test_hessian(self):
        Rs = 0.1
        x = np.linspace(Rs * 0.0001, Rs * 100, 10000)
        y = np.linspace(Rs * 0.001, Rs * 100, 10000)
        theta_Rs = 0.1

        tau = 10
        p = 0.7
        tau = 10
        p = 0.7

        r_trunc = Rs * tau
        r_core = p * Rs

        fxxcore, fxycore, fyycore = self.tcore.hessian(x, y, Rs, 1, r_trunc, r_core)
        fxx1, fxy1, fyy1 = self.tnfw.hessian(x, y, Rs, 1, r_trunc)
        fxx2, fxy2, fyy2 = self.tnfw.hessian(x, y, Rs, 1, r_core)

        np.testing.assert_almost_equal(fxxcore, fxx1 - fxx2)
        np.testing.assert_almost_equal(fxycore, fxy1 - fxy2)
        np.testing.assert_almost_equal(fyycore, fyy1 - fyy2)

    def test_density_2d(self):
        Rs = 0.1
        x = np.linspace(Rs * 0.0001, Rs * 100, 10000)
        y = np.linspace(Rs * 0.001, Rs * 100, 10000)
        theta_Rs = 0.1

        tau = 10
        p = 0.7

        r_trunc = Rs * tau
        r_core = p * Rs

        kappa_t = self.tnfw.density_2d(x, y, Rs, theta_Rs, r_trunc)
        kappa = self.tnfw.density_2d(x, y, Rs, theta_Rs, r_core)
        kappa = self.tcore.density_2d(x, y, Rs, theta_Rs, r_trunc, r_core)
        np.testing.assert_almost_equal(kappa, kappa_t-kappa, 5)

if __name__ == '__main__':
    pytest.main()
