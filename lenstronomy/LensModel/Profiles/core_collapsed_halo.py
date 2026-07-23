__author__ = "dgilman"

import numpy as np
from scipy.special import hyp2f1, beta
from scipy.interpolate import CubicSpline
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
__all__ = ["CoreCollapsedHalo"]

_MIN = 0.00000001


class CoreCollapsedHalo(LensProfileBase):

    profile_name = "CORE_COLLAPSED_HALO"
    param_names = ["center_x", "center_y", "Rs_inner", "Rs_outer", "alpha_Rs_inner", "alpha_Rs_outer",
                   "r_trunc", "gamma_inner", "gamma_outer"]
    lower_limit_default = {"Rs": 0, "alpha_Rs": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"Rs": 100, "alpha_Rs": 10, "center_x": 100, "center_y": 100}

    # class-level (process-wide) caches keyed on the slope pair (and grid
    # settings for the spline). Shared across every instance lenstronomy or
    # pyHalo creates, so each unique configuration is built at most once per
    # process regardless of how many CoreCollapsedHalo objects exist.
    _inner_const_cache = {}
    _inner_g_spline_cache = {}

    def __init__(self, interpol=True, num_interp_X=2000, max_interp_X=1000.0):
        """

        :param interpol: bool, if True, interpolates the inner profile's
         projected-mass function g() used for the deflection angles
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated
         (values outside [1/max_interp_X, max_interp_X] fall back to the exact evaluation)
        """
        self._profile_inner = PseudoDoublePowerlaw()
        self._profile_outer = TNFW()
        self._interpol = interpol
        self._num_interp_X = int(num_interp_X)
        self._max_interp_X = float(max_interp_X)
        super(CoreCollapsedHalo, self).__init__()

    def _inner_g_spline(self, gamma_inner, gamma_outer):
        """Cubic spline of the inner profile's projected-mass function
        ``_g(X)`` in log(X), tabulated once per (gamma_inner, gamma_outer).

        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: CubicSpline mapping log(X) -> _g(X)
        """
        key = (gamma_inner, gamma_outer, self._num_interp_X, self._max_interp_X)
        spline = self._inner_g_spline_cache.get(key)
        if spline is None:
            log_x = np.linspace(
                np.log(1.0 / self._max_interp_X),
                np.log(self._max_interp_X),
                self._num_interp_X,
            )
            gx = self._profile_inner._g(np.exp(log_x), gamma_inner, gamma_outer)
            spline = CubicSpline(log_x, gx, extrapolate=False)
            self._inner_g_spline_cache[key] = spline
        return spline

    def _inner_g_constants(self, gamma_inner, gamma_outer):
        """Position-independent constants entering the inner profile's projected
        mass function ``_g``, cached per (gamma_inner, gamma_outer).

        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: (n, g, beta_term_1, beta_term_2, gx1) where gx1 = _g(1.0)
        """
        key = (gamma_inner, gamma_outer)
        cached = self._inner_const_cache.get(key)
        if cached is None:
            g = gamma_inner
            n = gamma_outer
            if n == 3:
                n = 3.001  # for numerical stability (matches PseudoDoublePowerlaw)
            beta_term_1 = beta((n - 3) / 2, (3 - g) / 2)
            beta_term_2 = beta((n - 3) / 2, 1.5)
            # _g(1.0): X = 1 -> xi = 2, 1 / xi = 0.5
            gx1 = 0.5 * (
                beta_term_1
                - beta_term_2 * hyp2f1((n - 3) / 2, g / 2, n / 2, 0.5) * 2 ** ((3 - n) / 2)
            )
            cached = (n, g, beta_term_1, beta_term_2, gx1)
            self._inner_const_cache[key] = cached
        return cached

    def _inner_derivatives(self, x_, y_, Rs, alpha_Rs, gamma_inner, gamma_outer):
        """Deflection angles of the inner pseudo double power-law component,
        evaluated with cached slope-dependent constants so that only a single
        (unavoidable) hyp2f1 call over the coordinate array is performed.

        :param x_: x coordinate relative to the profile center
        :param y_: y coordinate relative to the profile center
        :param Rs: scale radius
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: deflection angle in x, deflection angle in y
        """
        n, g, beta_term_1, beta_term_2, gx1 = self._inner_g_constants(
            gamma_inner, gamma_outer
        )
        Rs = np.maximum(Rs, _MIN)
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        R = np.maximum(R, _MIN)
        X = R / Rs
        gx = self._inner_gx(X, n, g, beta_term_1, beta_term_2, gamma_inner, gamma_outer)
        # rho0 = alpha_Rs / (4 Rs^2 gx1); a = 4 rho0 Rs gx / X^2 -> collapses to:
        a = alpha_Rs * gx / (Rs * gx1 * X ** 2)
        return a * x_, a * y_

    def _inner_gx(self, X, n, g, beta_term_1, beta_term_2, gamma_inner, gamma_outer):
        """Evaluate the inner projected-mass function ``_g(X)``.

        Uses the cached cubic spline when ``interpol`` is enabled, evaluating the
        exact hyp2f1 form only for the (usually empty) set of points outside the
        interpolation grid. Falls back to the exact form entirely otherwise.
        """
        exact = lambda Xv: 0.5 * (
            beta_term_1
            - beta_term_2
            * hyp2f1((n - 3) / 2, g / 2, n / 2, 1 / (1 + Xv ** 2))
            * (1 + Xv ** 2) ** ((3 - n) / 2)
        )
        if not self._interpol:
            return exact(X)
        shape = np.shape(X)
        X_arr = np.atleast_1d(X).astype(float)
        gx = self._inner_g_spline(gamma_inner, gamma_outer)(np.log(X_arr))
        out = np.isnan(gx)  # points outside the tabulated range
        if np.any(out):
            gx[out] = exact(X_arr[out])
        if shape == ():
            return gx[0]
        return gx.reshape(shape)

    def split_kwargs(self, center_x, center_y, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer):
        """

        :param center_x:
        :param center_y:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param gamma_inner:
        :param gamma_outer:
        :return:
        """
        kwargs_inner = {'alpha_Rs': alpha_Rs_inner, 'center_x': center_x, 'center_y': center_y, 'Rs': Rs_inner,
                        'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer}
        kwargs_outer = {'alpha_Rs': alpha_Rs_outer, 'center_x': center_x, 'center_y': center_y, 'Rs': Rs_outer,
                        'r_trunc': r_trunc}
        return kwargs_inner, kwargs_outer

    def derivatives(self, x, y, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param center_x:
        :param center_y:
        :return:
        """
        f_x_inner, f_y_inner = self._inner_derivatives(x - center_x, y - center_y, Rs_inner,
                                                       alpha_Rs_inner, gamma_inner, gamma_outer)
        f_x_outer, f_y_outer = self._profile_outer.derivatives(x, y, Rs=Rs_outer, alpha_Rs=alpha_Rs_outer,
                                                              r_trunc=r_trunc, center_x=center_x, center_y=center_y)
        return f_x_inner + f_x_outer, f_y_inner + f_y_outer

    def hessian(self, x, y, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer,
                center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param Rs:
        :param alpha_Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        kwargs_inner, kwargs_outer = self.split_kwargs(center_x, center_y, Rs_inner, Rs_outer, alpha_Rs_inner,
                                                       alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer)
        f_xx_inner, f_xy_inner, f_yx_inner, f_yy_inner = self._profile_inner.hessian(x, y, **kwargs_inner)
        f_xx_outer, f_xy_outer, f_yx_outer, f_yy_outer = self._profile_outer.hessian(x, y, **kwargs_outer)
        return f_xx_inner + f_xx_outer, f_xy_inner + f_xy_outer, f_yx_inner + f_yx_outer, f_yy_inner + f_yy_outer

    def density_lens(self, R, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer):
        """

        :param R:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param gamma_inner:
        :param gamma_outer:
        :return:
        """
        density_inner = self._profile_inner.density_lens(R, Rs_inner, alpha_Rs_inner, gamma_inner, gamma_outer)
        rho0 = self._profile_outer.alpha2rho0(alpha_Rs_outer, Rs_outer)
        density_outer = self._profile_outer.density(R, Rs_outer, rho0, r_trunc)
        return density_inner + density_outer

    def density_2d_lens(self, x, y, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner,
                        gamma_outer, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param gamma_inner:
        :param gamma_outer:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self._profile_inner.alpha2rho0(alpha_Rs_inner, Rs_inner, gamma_inner, gamma_outer)
        density_2d_inner = self._profile_inner.density_2d(x, y, Rs_inner, rho0,
                                                          gamma_inner, gamma_outer, center_x, center_y)
        rho0 = self._profile_outer.alpha2rho0(alpha_Rs_outer, Rs_outer)
        density_2d_outer = self._profile_outer.density_2d(x, y, Rs_outer, rho0, r_trunc, center_x, center_y)
        return density_2d_inner + density_2d_outer

    def mass_3d_lens(self, R, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer):
        """

        :param R:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param gamma_inner:
        :param gamma_outer:
        :return:
        """
        mass_3d_inner = self._profile_inner.mass_3d_lens(R, Rs_inner, alpha_Rs_inner, gamma_inner, gamma_outer)
        rho0 = self._profile_outer.alpha2rho0(alpha_Rs_outer, Rs_outer)
        mass_3d_outer = self._profile_outer.mass_3d(R, Rs_outer, rho0, r_trunc/Rs_outer)
        return mass_3d_outer + mass_3d_inner

    def mass_2d_lens(self, R, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer):
        """

        :param R:
        :param Rs_inner:
        :param Rs_outer:
        :param alpha_Rs_inner:
        :param alpha_Rs_outer:
        :param r_trunc:
        :param gamma_inner:
        :param gamma_outer:
        :return:
        """
        rho0 = self._profile_inner.alpha2rho0(alpha_Rs_inner, Rs_inner, gamma_inner, gamma_outer)
        mass_2d_inner = self._profile_inner.mass_2d(R, Rs_inner, rho0, gamma_inner, gamma_outer)
        rho0 = self._profile_outer.alpha2rho0(alpha_Rs_outer, Rs_outer)
        mass_2d_outer = self._profile_outer.mass_2d(R, Rs_outer, rho0, r_trunc)
        return mass_2d_outer + mass_2d_inner
