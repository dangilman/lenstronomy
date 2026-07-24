__author__ = "dangilman"

from lenstronomy.LensModel.Profiles.king import King, KingRc

import numpy as np
from scipy.integrate import quad
import pytest
import numpy.testing as npt


class TestKingRc(object):
    def setup_method(self):
        self.profile = KingRc()

    def test_origin(self):
        x = 0.0
        y = 0.0
        sigma0 = 1.0
        r_core = 0.1
        c = 1.5
        alpha_x, alpha_y = self.profile.derivatives(x, y, sigma0, r_core, c)
        npt.assert_almost_equal(alpha_x, 0.0)
        npt.assert_almost_equal(alpha_y, 0.0)

        fxx, fxy, fyx, fyy = self.profile.hessian(x, y, sigma0, r_core, c)
        kappa = self.profile.density_2d(x, y, sigma0, r_core, c)
        npt.assert_almost_equal(fxx, kappa)
        npt.assert_almost_equal(fyy, kappa)
        npt.assert_almost_equal(fxy, 0.0)
        npt.assert_almost_equal(fyx, 0.0)

    def test_central_convergence(self):
        # sigma0 is the central convergence kappa(0)
        sigma0, r_core, c = 0.7, 0.2, 1.3
        kappa0 = self.profile.density_2d(0.0, 0.0, sigma0, r_core, c)
        npt.assert_almost_equal(kappa0, sigma0)

    def test_truncation(self):
        # convergence is zero beyond the tidal radius; enclosed mass is constant there
        sigma0, r_core, c = 1.0, 0.3, 1.2
        r_t = r_core * 10 ** c
        npt.assert_almost_equal(self.profile.density_2d(1.001 * r_t, 0.0, sigma0, r_core, c), 0.0)
        m_in = self.profile.mass_2d_lens(r_t, sigma0, r_core, c)
        m_out = self.profile.mass_2d_lens(5 * r_t, sigma0, r_core, c)
        npt.assert_almost_equal(m_out, m_in)

    def test_mass_2d(self):
        # projected enclosed mass matches the numerical surface-density integral
        sigma0, r_core = 1.0, 0.05
        for c in [0.8, 1.2, 2.0]:
            R = 0.5 * r_core * 10 ** c  # inside the tidal radius
            args = (sigma0, r_core, c)
            mass_numerical = quad(self._mass_integrand2d, 0, R, args=args)[0]
            mass_analytic = self.profile.mass_2d_lens(R, sigma0, r_core, c)
            npt.assert_almost_equal(mass_analytic, mass_numerical, decimal=6)

    def test_derivatives(self):
        # analytic deflection vs numerical gradient of the potential
        sigma0, r_core, c = 1.2, 0.3, 1.4
        diff = 1e-6
        for x, y in [(0.9, 0.6), (2.0, -1.3), (0.05, 0.02)]:
            f_x, f_y = self.profile.derivatives(x, y, sigma0, r_core, c)
            f_dx = (self.profile.function(x + diff, y, sigma0, r_core, c)
                    - self.profile.function(x - diff, y, sigma0, r_core, c)) / (2 * diff)
            f_dy = (self.profile.function(x, y + diff, sigma0, r_core, c)
                    - self.profile.function(x, y - diff, sigma0, r_core, c)) / (2 * diff)
            npt.assert_almost_equal(f_x, f_dx, decimal=5)
            npt.assert_almost_equal(f_y, f_dy, decimal=5)

    def test_hessian(self):
        # analytic Hessian vs numerical derivatives of the deflection angles
        sigma0, r_core, c = 1.2, 0.3, 1.4
        diff = 1e-6
        for x, y in [(0.9, 0.6), (2.0, -1.3)]:
            f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, r_core, c)
            f_x_px, f_y_px = self.profile.derivatives(x + diff, y, sigma0, r_core, c)
            f_x_mx, f_y_mx = self.profile.derivatives(x - diff, y, sigma0, r_core, c)
            f_x_py, f_y_py = self.profile.derivatives(x, y + diff, sigma0, r_core, c)
            f_x_my, f_y_my = self.profile.derivatives(x, y - diff, sigma0, r_core, c)
            npt.assert_almost_equal(f_xx, (f_x_px - f_x_mx) / (2 * diff), decimal=5)
            npt.assert_almost_equal(f_yy, (f_y_py - f_y_my) / (2 * diff), decimal=5)
            npt.assert_almost_equal(f_xy, (f_x_py - f_x_my) / (2 * diff), decimal=5)
            npt.assert_almost_equal(f_yx, (f_y_px - f_y_mx) / (2 * diff), decimal=5)

    def test_convergence(self):
        # kappa recovered from the trace of the Hessian, kappa = (f_xx + f_yy) / 2
        sigma0, r_core, c = 0.9, 0.25, 1.6
        for x, y in [(0.3, 0.2), (1.0, 0.5), (3.0, 1.0)]:
            f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, r_core, c)
            kappa = self.profile.density_2d(x, y, sigma0, r_core, c)
            npt.assert_almost_equal(0.5 * (f_xx + f_yy), kappa)

    def test_sigma0_from_mass_2d(self):
        # round trip: sigma0 -> total projected mass -> sigma0
        sigma0, r_core, c = 0.8, 0.5, 1.3
        m_total = self.profile.mass_2d_lens(10 * r_core * 10 ** c, sigma0, r_core, c)
        sigma0_out = self.profile.sigma0_from_mass_2d(m_total, r_core, c)
        npt.assert_almost_equal(sigma0_out, sigma0)

    def _mass_integrand2d(self, r, sigma0, r_core, c):
        return 2 * np.pi * r * self.profile.density_2d(r, 0, sigma0, r_core, c)


class TestKing(object):
    def setup_method(self):
        self.profile = King()
        self.kingrc = KingRc()

    def test_half_mass_radius(self):
        # r_h is the projected half-mass radius, independent of the normalization
        sigma0, r_h = 1.0, 3.0
        for c in [0.5, 1.5, 3.0]:
            r_t = self.profile._r_core(r_h, c) * 10 ** c
            m_half = self.profile.mass_2d_lens(r_h, sigma0, r_h, c)
            m_total = self.profile.mass_2d_lens(5 * r_t, sigma0, r_h, c)
            npt.assert_almost_equal(m_half / m_total, 0.5, decimal=4)

    def test_equivalence_to_kingrc(self):
        # King(r_h, c) equals KingRc at the mapped core radius r_c = r_h / f(c)
        sigma0, r_h, c = 1.1, 2.5, 1.4
        r_core = self.profile._r_core(r_h, c)
        x, y = 0.8, 0.5
        npt.assert_almost_equal(self.profile.function(x, y, sigma0, r_h, c),
                                self.kingrc.function(x, y, sigma0, r_core, c))
        fx_h, fy_h = self.profile.derivatives(x, y, sigma0, r_h, c)
        fx_c, fy_c = self.kingrc.derivatives(x, y, sigma0, r_core, c)
        npt.assert_almost_equal(fx_h, fx_c)
        npt.assert_almost_equal(fy_h, fy_c)
        h_h = self.profile.hessian(x, y, sigma0, r_h, c)
        h_c = self.kingrc.hessian(x, y, sigma0, r_core, c)
        npt.assert_almost_equal(h_h, h_c)

    def test_derivatives(self):
        # deflection consistent with the numerical gradient of the potential
        sigma0, r_h, c = 1.0, 2.0, 1.5
        diff = 1e-6
        x, y = 0.7, 0.4
        f_x, f_y = self.profile.derivatives(x, y, sigma0, r_h, c)
        f_dx = (self.profile.function(x + diff, y, sigma0, r_h, c)
                - self.profile.function(x - diff, y, sigma0, r_h, c)) / (2 * diff)
        f_dy = (self.profile.function(x, y + diff, sigma0, r_h, c)
                - self.profile.function(x, y - diff, sigma0, r_h, c)) / (2 * diff)
        npt.assert_almost_equal(f_x, f_dx, decimal=5)
        npt.assert_almost_equal(f_y, f_dy, decimal=5)


if __name__ == "__main__":
    pytest.main()
