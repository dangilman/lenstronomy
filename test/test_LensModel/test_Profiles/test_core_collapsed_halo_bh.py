__author__ = "dangilman"

import numpy.testing as npt
import pytest
import numpy as np
from lenstronomy.LensModel.Profiles.core_collapsed_halo_bh import CoreCollapsedHaloBH
from lenstronomy.LensModel.Profiles.point_mass import PointMass
from lenstronomy.LensModel.Profiles.tnfw import TNFW


class TestCoreCollapsedHaloBH(object):

    def setup_method(self):

        self.profile = CoreCollapsedHaloBH()
        self.inner = PointMass()
        self.outer = TNFW()
        self.kwargs_inner = {
            "theta_E": 1.0,
            "center_x": 0.5,
            "center_y": 0.1,
        }
        self.kwargs_outer = {
            "alpha_Rs": 1.0,
            "center_x": 0.5,
            "center_y": 0.1,
            "Rs": 2.0,
            "r_trunc": 5.0
        }
        self.kwargs_cc = {
            "theta_E_inner": self.kwargs_inner["theta_E"],
            "Rs_outer": self.kwargs_outer["Rs"],
            "alpha_Rs_outer": self.kwargs_outer["alpha_Rs"],
            "r_trunc": self.kwargs_outer["r_trunc"],
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

if __name__ == "__main__":
    pytest.main()