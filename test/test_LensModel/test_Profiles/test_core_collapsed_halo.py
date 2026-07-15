__author__ = "dangilman"

import numpy.testing as npt
import pytest
import numpy as np
from lenstronomy.LensModel.Profiles.core_collapsed_halo import CoreCollapsedHalo
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
from lenstronomy.LensModel.Profiles.tnfw import TNFW


class TestCoreCollapsedHalo(object):
    """Test TestEPL_MULTIPOLE_M1M3M4 vs TestEPL_MULTIPOLE_M3M4 with m1 term = 0."""

    def setup_method(self):
        # param_names = ["center_x", "center_y", "Rs_inner", "Rs_outer", "alpha_Rs_inner", "alpha_Rs_outer",
        #                    "r_trunc", "gamma_inner", "gamma_outer"]
        self.profile = CoreCollapsedHalo()
        self.inner = PseudoDoublePowerlaw()
        self.outer = TNFW()
        self.kwargs_inner = {
            "alpha_Rs": 1.0,
            "center_x": 0.5,
            "center_y": 0.1,
            "Rs": 1.0,
            "gamma_inner": 2.5,
            "gamma_outer": 5.0,
        }
        self.kwargs_outer = {
            "alpha_Rs": 1.0,
            "center_x": 0.5,
            "center_y": 0.1,
            "Rs": 2.0,
            "r_trunc": 5.0
        }
        self.kwargs_cc = {
            "Rs_inner": self.kwargs_inner["Rs"],
            "Rs_outer": self.kwargs_outer["Rs"],
            "alpha_Rs_inner": self.kwargs_inner["alpha_Rs"],
            "alpha_Rs_outer": self.kwargs_outer["alpha_Rs"],
            "r_trunc": self.kwargs_outer["r_trunc"],
            "gamma_inner": self.kwargs_inner["gamma_inner"],
            "gamma_outer": self.kwargs_inner["gamma_outer"],
            "center_x": self.kwargs_inner["center_x"],
            "center_y": self.kwargs_inner["center_y"],
        }

    def test_derivatives(self):

        x, y = 1.5, -1.5
        fx1, fy1 = self.inner.derivatives(x, y, **self.kwargs_inner)
        fx2, fy2 = self.outer.derivatives(x, y, **self.kwargs_outer)
        fx, fy = self.profile.derivatives(x, y, **self.kwargs_cc)
        npt.assert_almost_equal(fx1 + fx2, fx)
        npt.assert_almost_equal(fy1 + fy2, fy)

        x, y = 1.5, np.linspace(-1, 1, 10)
        fx1, fy1 = self.inner.derivatives(x, y, **self.kwargs_inner)
        fx2, fy2 = self.outer.derivatives(x, y, **self.kwargs_outer)
        fx, fy = self.profile.derivatives(x, y, **self.kwargs_cc)
        npt.assert_almost_equal(fx1 + fx2, fx)
        npt.assert_almost_equal(fy1 + fy2, fy)

    def test_hessian(self):

        x, y = 1.5, -1.5
        fxx1, fxy1, fyx1, fyy1 = self.inner.hessian(x, y, **self.kwargs_inner)
        fxx2, fxy2, fyx2, fyy2 = self.outer.hessian(x, y, **self.kwargs_outer)
        fxx, fxy, fyx, fyy = self.profile.hessian(x, y, **self.kwargs_cc)
        npt.assert_almost_equal(fxx1 + fxx2, fxx)
        npt.assert_almost_equal(fyy1 + fyy2, fyy)
        npt.assert_almost_equal(fxy1 + fxy2, fxy)
        npt.assert_almost_equal(fyx1 + fyx2, fyx)

        x, y = 1.5, np.linspace(-1, 1, 10)
        fxx1, fxy1, fyx1, fyy1 = self.inner.hessian(x, y, **self.kwargs_inner)
        fxx2, fxy2, fyx2, fyy2 = self.outer.hessian(x, y, **self.kwargs_outer)
        fxx, fxy, fyx, fyy = self.profile.hessian(x, y, **self.kwargs_cc)
        npt.assert_almost_equal(fxx1 + fxx2, fxx)
        npt.assert_almost_equal(fyy1 + fyy2, fyy)
        npt.assert_almost_equal(fxy1 + fxy2, fxy)
        npt.assert_almost_equal(fyx1 + fyx2, fyx)

    def test_density(self):

        x, y = 1.5, -1.5
        density1 = self.inner.density_lens(np.hypot(x,y), self.kwargs_inner['Rs'], self.kwargs_inner['alpha_Rs'],
                                           self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.density(np.hypot(x,y), self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.density_lens(np.hypot(x,y), self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

        x, y = 1.5, np.linspace(-1, 1, 10)
        density1 = self.inner.density_lens(np.hypot(x, y), self.kwargs_inner['Rs'], self.kwargs_inner['alpha_Rs'],
                                           self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.density(np.hypot(x, y), self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.density_lens(np.hypot(x, y), self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

    def test_density_2d(self):

        x, y = 1.5, -1.5
        rho0 = self.inner.alpha2rho0(self.kwargs_inner['alpha_Rs'], self.kwargs_inner['Rs'],
                                     self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        density1 = self.inner.density_2d(x, y, self.kwargs_inner['Rs'], rho0, self.kwargs_inner['gamma_inner'],
                                         self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.density_2d(x, y, self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.density_2d_lens(x, y, self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

        x, y = 1.5, np.linspace(-1, 1, 10)
        rho0 = self.inner.alpha2rho0(self.kwargs_inner['alpha_Rs'], self.kwargs_inner['Rs'],
                                     self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        density1 = self.inner.density_2d(x, y, self.kwargs_inner['Rs'], rho0, self.kwargs_inner['gamma_inner'],
                                         self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.density_2d(x, y, self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.density_2d_lens(x, y, self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

    def test_mass_2d(self):

        x, y = 1.5, -1.5
        rho0 = self.inner.alpha2rho0(self.kwargs_inner['alpha_Rs'], self.kwargs_inner['Rs'],
                                     self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        density1 = self.inner.mass_2d(np.hypot(x,y), self.kwargs_inner['Rs'], rho0, self.kwargs_inner['gamma_inner'],
                                         self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.mass_2d(np.hypot(x,y), self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.mass_2d_lens(np.hypot(x,y), self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

        x, y = 1.5, np.linspace(-1, 1, 10)
        rho0 = self.inner.alpha2rho0(self.kwargs_inner['alpha_Rs'], self.kwargs_inner['Rs'],
                                     self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        density1 = self.inner.mass_2d(np.hypot(x, y), self.kwargs_inner['Rs'], rho0, self.kwargs_inner['gamma_inner'],
                                      self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        density2 = self.outer.mass_2d(np.hypot(x, y), self.kwargs_outer['Rs'], rho0, self.kwargs_outer['r_trunc'])
        density = self.profile.mass_2d_lens(np.hypot(x, y), self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                       self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                       self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                       self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(density1 + density2, density)

    def test_mass_3d(self):

        x, y = 1.5, -1.5
        rho0 = self.inner.alpha2rho0(self.kwargs_inner['alpha_Rs'], self.kwargs_inner['Rs'],
                                     self.kwargs_inner['gamma_inner'], self.kwargs_inner['gamma_outer'])
        mass1 = self.inner.mass_3d(np.hypot(x,y), self.kwargs_inner['Rs'], rho0, self.kwargs_inner['gamma_inner'],
                                         self.kwargs_inner['gamma_outer'])
        rho0 = self.outer.alpha2rho0(self.kwargs_outer['alpha_Rs'], self.kwargs_outer['Rs'])
        mass2 = self.outer.mass_3d(np.hypot(x,y), self.kwargs_outer['Rs'], rho0,
                                   self.kwargs_outer['r_trunc']/self.kwargs_outer['Rs'])
        mass = self.profile.mass_3d_lens(np.hypot(x,y), self.kwargs_cc['Rs_inner'], self.kwargs_cc['Rs_outer'],
                                            self.kwargs_cc['alpha_Rs_inner'], self.kwargs_cc['alpha_Rs_outer'],
                                            self.kwargs_cc['r_trunc'], self.kwargs_cc['gamma_inner'],
                                            self.kwargs_cc['gamma_outer'])
        npt.assert_almost_equal(mass1 + mass2, mass)

if __name__ == "__main__":
    pytest.main()