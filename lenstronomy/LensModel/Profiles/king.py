__author__ = "dangilman"

import numpy as np
from scipy.special import spence
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["King"]


class KingRc(LensProfileBase):
    r"""Empirical King (1962) profile as a circular lens model, parameterized by the
    core radius and the concentration.

    The projected (surface-mass) density follows the King (1962) empirical form

    .. math::
        \kappa(R) = \sigma_0 \,
        \frac{\left[(1 + (R/r_c)^2)^{-1/2} - a\right]^2}{(1 - a)^2},
        \qquad R \le r_t,

    and :math:`\kappa = 0` for :math:`R > r_t`, where

    .. math::
        a \equiv (1 + (r_t/r_c)^2)^{-1/2}, \qquad c \equiv \log_{10}(r_t/r_c).

    The subtraction of :math:`a` forces the density smoothly to zero at the tidal
    radius :math:`r_t`. ``sigma0`` is the central convergence :math:`\kappa(0)`. The
    model is parameterized by the core radius ``r_core`` (:math:`r_c`) and the King
    concentration ``c``; the tidal radius is :math:`r_t = r_c \, 10^{c}`.

    The projected enclosed mass is

    .. math::
        m_{2D}(<R) = 2\pi A r_c^2 \,
        \Big[ \tfrac12 \ln(1+x^2) - 2a(\sqrt{1+x^2}-1)
              + \tfrac{a^2}{2} x^2 \Big],
        \quad x = R/r_c,\; A = \sigma_0/(1-a)^2,

    (evaluated at :math:`x_t = r_t/r_c` for :math:`R > r_t`). This is a projected
    (surface) prescription; a 3D density is not defined for the King (1962) form and
    is not implemented.
    """

    param_names = ["sigma0", "r_core", "c", "center_x", "center_y"]
    lower_limit_default = {"sigma0": 0, "r_core": 1e-6, "c": 0.3, "center_x": -100, "center_y": -100}
    upper_limit_default = {"sigma0": 1e12, "r_core": 100, "c": 6.0, "center_x": 100, "center_y": 100}

    _s = 1e-9  # minimum radius (in units of r_core) used to avoid division by zero

    @staticmethod
    def _a(c):
        """Dimensionless truncation constant :math:`a = (1 + (r_t/r_c)^2)^{-1/2}`, with
        :math:`r_t/r_c = 10^c`.

        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: truncation constant a
        """
        return 1.0 / np.sqrt(1.0 + 10.0 ** (2.0 * c))

    @staticmethod
    def _F(x, a):
        """Dimensionless enclosed-mass function such that
        :math:`m_{2D} = 2\\pi A r_c^2 F(x)`,

        .. math::
            F(x) = \\tfrac12 \\ln(1+x^2) - 2a(\\sqrt{1+x^2}-1) + \\tfrac{a^2}{2} x^2 .

        :param x: dimensionless projected radius :math:`x = R/r_c`
        :param a: truncation constant from :func:`_a`
        :return: value of F(x)
        """
        s = np.sqrt(1.0 + x ** 2)
        return 0.5 * np.log(1.0 + x ** 2) - 2.0 * a * (s - 1.0) + 0.5 * a ** 2 * x ** 2

    @staticmethod
    def _G(x, a):
        """Antiderivative used for the lensing potential, defined by :math:`G'(x) =
        F(x)/x`,

        .. math::
            G(x) = -\\tfrac14 \\mathrm{Li}_2(-x^2)
                   - 2a\\left[\\sqrt{1+x^2} - \\ln(1+\\sqrt{1+x^2})\\right]
                   + \\tfrac{a^2}{4} x^2 .

        Uses :math:`\\mathrm{Li}_2(-x^2) = \\mathrm{spence}(1 + x^2)` (SciPy convention
        :math:`\\mathrm{Li}_2(w) = \\mathrm{spence}(1-w)`).

        :param x: dimensionless projected radius :math:`x = R/r_c`
        :param a: truncation constant from :func:`_a`
        :return: value of G(x)
        """
        s = np.sqrt(1.0 + x ** 2)
        dilog = spence(1.0 + x ** 2)  # = Li_2(-x^2)
        return -0.25 * dilog - 2.0 * a * (s - np.log(1.0 + s)) + 0.25 * a ** 2 * x ** 2

    def density_2d(self, x, y, sigma0, r_core, c, center_x=0, center_y=0):
        """Projected (surface) convergence :math:`\\kappa(R)`, zero beyond the tidal
        radius.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: convergence kappa at the projected radius of (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        a = self._a(c)
        r_t = r_core * 10.0 ** c
        xx = r / r_core
        shape = ((1.0 + xx ** 2) ** -0.5 - a) ** 2
        kappa = sigma0 * shape / (1.0 - a) ** 2
        return np.where(r > r_t, 0.0, kappa)

    def mass_2d_lens(self, r, sigma0, r_core, c):
        """Projected enclosed mass in convergence units, :math:`m_{2D}(<r) = 2\\pi
        \\int_0^r \\kappa(r') r' dr'`. All mass is enclosed beyond the tidal radius.

        :param r: projected radius (angular units)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: projected mass enclosed within r
        """
        a = self._a(c)
        A = sigma0 / (1.0 - a) ** 2
        x_t = 10.0 ** c
        x_eff = np.minimum(r / r_core, x_t)  # all mass enclosed beyond r_t
        return 2.0 * np.pi * A * r_core ** 2 * self._F(x_eff, a)

    def alpha(self, r, sigma0, r_core, c):
        """Radial deflection angle, :math:`\\alpha(r) = m_{2D}(<r) / (\\pi r)`.

        :param r: projected radius (angular units)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: radial deflection angle at r
        """
        r_safe = np.maximum(r, self._s * r_core)
        return self.mass_2d_lens(r_safe, sigma0, r_core, c) / (np.pi * r_safe)

    def function(self, x, y, sigma0, r_core, c, center_x=0, center_y=0):
        """Lensing potential (analytic). Inside :math:`r_t` it follows
        :math:`\\psi = 2 A r_c^2 [G(x) - G(0)]`; outside :math:`r_t` it continues as a
        point mass.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: lensing potential at (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.maximum(np.sqrt(x_ ** 2 + y_ ** 2), self._s * r_core)
        a = self._a(c)
        A = sigma0 / (1.0 - a) ** 2
        r_t = r_core * 10.0 ** c
        x_t = 10.0 ** c
        G0 = self._G(0.0, a)
        psi_in = 2.0 * A * r_core ** 2 * (self._G(r / r_core, a) - G0)
        m_tot = 2.0 * np.pi * A * r_core ** 2 * self._F(x_t, a)
        psi_t = 2.0 * A * r_core ** 2 * (self._G(x_t, a) - G0)
        psi_out = psi_t + (m_tot / np.pi) * np.log(r / r_t)
        return np.where(r > r_t, psi_out, psi_in)

    def derivatives(self, x, y, sigma0, r_core, c, center_x=0, center_y=0):
        """Deflection angles :math:`\\alpha_x, \\alpha_y` (first derivatives of the
        lensing potential).

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: deflection angle in x, deflection angle in y
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.maximum(np.sqrt(x_ ** 2 + y_ ** 2), self._s * r_core)
        alpha_r = self.alpha(r, sigma0, r_core, c)
        return alpha_r * x_ / r, alpha_r * y_ / r

    def hessian(self, x, y, sigma0, r_core, c, center_x=0, center_y=0):
        """Second derivatives of the lensing potential, :math:`f_{xx}, f_{xy}, f_{yx},
        f_{yy}`.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: Hessian components f_xx, f_xy, f_yx, f_yy
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.maximum(np.sqrt(x_ ** 2 + y_ ** 2), self._s * r_core)
        kappa = self.density_2d(x_, y_, sigma0, r_core, c)
        m2d = self.mass_2d_lens(r, sigma0, r_core, c) / np.pi
        gamma_tot = m2d / r ** 2 - kappa
        cos_2phi = (y_ ** 2 - x_ ** 2) / r ** 2
        sin_2phi = -2.0 * x_ * y_ / r ** 2
        f_xx = kappa + cos_2phi * gamma_tot
        f_yy = kappa - cos_2phi * gamma_tot
        f_xy = sin_2phi * gamma_tot
        return f_xx, f_xy, f_xy, f_yy

    def sigma0_from_mass_2d(self, m_2d_total, r_core, c):
        """Central convergence ``sigma0`` that yields a desired total projected mass.

        The mass is in convergence units, i.e. physical mass divided by the critical
        surface density for lensing in matching angular units.

        :param m_2d_total: total projected mass (convergence units)
        :param r_core: King core radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: central convergence sigma0
        """
        a = self._a(c)
        x_t = 10.0 ** c
        norm = 2.0 * np.pi * r_core ** 2 * self._F(x_t, a) / (1.0 - a) ** 2
        return m_2d_total / norm


class King(KingRc):
    r"""King (1962) lens profile parameterized by the projected HALF-MASS radius
    ``r_h`` and the concentration ``c = log10(r_t / r_c)``, instead of the core radius
    :math:`r_c`.

    Observations of globular clusters constrain the (projected) half-light / half-mass
    radius robustly and roughly mass-independently, whereas the King core radius is a
    derived fit quantity. Parameterizing by ``(r_h, c)`` therefore separates the
    well-measured size from the structural concentration.

    Internally it maps to the core radius via :math:`r_c = r_h / f(c)`, where
    :math:`f(c) = R_h / r_c` is the dimensionless projected half-mass radius in units
    of the core radius. :math:`f(c)` depends only on concentration and is precomputed
    once (by inverting the enclosed-mass profile) and interpolated.
    """

    param_names = ["sigma0", "r_h", "c", "center_x", "center_y"]
    lower_limit_default = {"sigma0": 0, "r_h": 1e-6, "c": 0.25, "center_x": -100, "center_y": -100}
    upper_limit_default = {"sigma0": 1e12, "r_h": 100, "c": 6.0, "center_x": 100, "center_y": 100}

    # precomputed f(c) = R_h / r_c grid (depends only on c)
    _C_GRID = np.linspace(0.25, 6.0, 200)
    _F_OF_C = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # delegate computation to a pure KingRc instance to avoid re-dispatching the
        # KingRc helpers back through the r_h -> r_core conversion
        self._king = KingRc()

    @classmethod
    def _build_fc(cls):
        """Tabulate and interpolate :math:`f(c) = R_h / r_c` on ``_C_GRID`` by solving
        :math:`m_{2D}(<R_h) = \\tfrac12 m_{2D}(<r_t)` at unit core radius for each c.

        :return: None (populates the class-level interpolator ``_F_OF_C``)
        """
        base = KingRc()

        def f_of_c(c):
            r_t = 10.0 ** c
            Mtot = base.mass_2d_lens(r_t, 1.0, 1.0, c)
            g = lambda R: base.mass_2d_lens(R, 1.0, 1.0, c) - 0.5 * Mtot
            return brentq(g, 1e-4 * r_t, r_t)  # r_core = 1 -> R_h in units of r_c

        vals = np.array([f_of_c(c) for c in cls._C_GRID])
        cls._F_OF_C = interp1d(cls._C_GRID, vals, kind="cubic",
                               bounds_error=False, fill_value=(vals[0], vals[-1]))

    @classmethod
    def f_of_c(cls, c):
        """Projected half-mass radius in units of the core radius, :math:`R_h / r_c`,
        interpolated over concentration.

        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: dimensionless ratio R_h / r_c
        """
        if cls._F_OF_C is None:
            cls._build_fc()
        return cls._F_OF_C(c)

    def _r_core(self, r_h, c):
        """Core radius implied by a half-mass radius and concentration,
        :math:`r_c = r_h / f(c)`.

        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: King core radius r_core
        """
        return r_h / self.f_of_c(c)

    def function(self, x, y, sigma0, r_h, c, center_x=0, center_y=0):
        """Lensing potential; see :meth:`KingRc.function`.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: lensing potential at (x, y)
        """
        return self._king.function(x, y, sigma0, self._r_core(r_h, c), c, center_x, center_y)

    def derivatives(self, x, y, sigma0, r_h, c, center_x=0, center_y=0):
        """Deflection angles; see :meth:`KingRc.derivatives`.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: deflection angle in x, deflection angle in y
        """
        return self._king.derivatives(x, y, sigma0, self._r_core(r_h, c), c, center_x, center_y)

    def hessian(self, x, y, sigma0, r_h, c, center_x=0, center_y=0):
        """Second derivatives of the lensing potential; see :meth:`KingRc.hessian`.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: Hessian components f_xx, f_xy, f_yx, f_yy
        """
        return self._king.hessian(x, y, sigma0, self._r_core(r_h, c), c, center_x, center_y)

    def density_2d(self, x, y, sigma0, r_h, c, center_x=0, center_y=0):
        """Projected convergence; see :meth:`KingRc.density_2d`.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :param center_x: center of the profile (angular units)
        :param center_y: center of the profile (angular units)
        :return: convergence kappa at the projected radius of (x, y)
        """
        return self._king.density_2d(x, y, sigma0, self._r_core(r_h, c), c, center_x, center_y)

    def mass_2d_lens(self, r, sigma0, r_h, c):
        """Projected enclosed mass; see :meth:`KingRc.mass_2d_lens`.

        :param r: projected radius (angular units)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: projected mass enclosed within r
        """
        return self._king.mass_2d_lens(r, sigma0, self._r_core(r_h, c), c)

    def alpha(self, r, sigma0, r_h, c):
        """Radial deflection angle; see :meth:`KingRc.alpha`.

        :param r: projected radius (angular units)
        :param sigma0: central convergence :math:`\\kappa(0)`
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: radial deflection angle at r
        """
        return self._king.alpha(r, sigma0, self._r_core(r_h, c), c)

    def sigma0_from_mass_2d(self, m_2d_total, r_h, c):
        """Central convergence for a desired total projected mass; see
        :meth:`KingRc.sigma0_from_mass_2d`.

        :param m_2d_total: total projected mass (convergence units)
        :param r_h: projected half-mass radius (angular units)
        :param c: King concentration :math:`c = \\log_{10}(r_t/r_c)`
        :return: central convergence sigma0
        """
        return self._king.sigma0_from_mass_2d(m_2d_total, self._r_core(r_h, c), c)
