__author__ = "dgilman"

"""Elliptical power law (free logarithmic slope gamma) with radially
varying axis ratio (ellipticity gradient) and position angle (isodensity
twist).

Self-contained lenstronomy profile module. To install into lenstronomy:

1. Copy this file to lenstronomy/LensModel/Profiles/epl_qgrad.py and place
   the precomputed table 'epl_qgrad_table.npz' in the same directory (or
   point to it with profile_kwargs {'table_path': ...} or the environment
   variable EPL_QGRAD_TABLE).
2. In lenstronomy/LensModel/profile_list_base.py add the strings
   "EPL_QGRAD" and "EPL_QGRAD_ELL" to _SUPPORTED_MODELS and insert (in
   alphabetical position in lens_class):

    elif lens_type == "EPL_QGRAD":
        from lenstronomy.LensModel.Profiles.epl_qgrad import EPLQGrad
        return EPLQGrad(**profile_kwargs)
    elif lens_type == "EPL_QGRAD_ELL":
        from lenstronomy.LensModel.Profiles.epl_qgrad import EPLQGradEllipse
        return EPLQGradEllipse(**profile_kwargs)

Model (intermediate-axis convention; dq = dphi = 0, gamma = 2 is exactly
the SIE with the same theta_E; dq = dphi = 0 is the EPL):

    kappa(x, y) = (3 - gamma)/2 * (theta_E / m)^(gamma - 1)
    q(m) u^2 + v^2 / q(m) = m^2,   (u + iv) = exp(-i psi(m)) (x' + iy')
    q(m)   = clip(q0 + dq * ln(m / theta_E), 0.2, 1.0)
    psi(m) = dphi * ln(m / theta_E)

where (x', y') is the frame rotated by phi_G. q0 is the axis ratio of the
isodensity contour at the Einstein radius, dq the change in axis ratio per
e-fold in radius, dphi the isodensity twist in radians per e-fold.

The deflection was computed exactly by decomposing the convergence into
uniform-density solid ellipses (2D homoeoid theorem; Bourassa & Kantowski
1975; Schramm 1990), which reduces it to two smooth 1D integrals evaluated
with Gauss-Legendre quadrature to <= 7e-7 relative error. The table stores
the complex ratio g = alpha*_model / (alpha*_SIE(q0) * R^(2 - gamma)) on
a (log10 R, phi, q0, dq, dphi, gamma) grid; at evaluation time g is
interpolated and multiplied by the closed-form SIE deflection and the exact
radial power law, so the isothermal SIE limit is exact and the dominant
radial/angular structure is not interpolation-limited.

Interpolation: the four parameter dimensions are contracted with a local
4-point Lagrange tensor cubic (cached per parameter tuple; parameters are
fixed while positions vary during ray tracing), positions are evaluated
through a bicubic spline in (log10 R, phi) -- the deflection field is C^2 in
sky coordinates, giving smooth hessians and magnifications
(|delta mu / mu| ~ 1e-4 at quad-image magnifications).

Validity ranges (table): gamma in [1.6, 2.5], q0 in [0.3, 1.0],
|dq| <= 0.2, |dphi| <= 0.2, R in [0.01, 10] theta_E (clamped to edge
values outside in log R; the log R grid is densest at 0.25 - 4 theta_E).
"""

import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["EPLQGrad", "EPLQGradEllipse", "EPL_QGRAD_MULTIPOLE_M1M3M4"]

_DEFAULT_TABLE = os.environ.get(
    "EPL_QGRAD_TABLE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "epl_qgrad_table.npz"))


def _sqrt_branch(w, c):
    """sqrt(w^2 - c^2), branch cut on the focal segment [-c, c], s ~ w at
    infinity (product of principal square roots)."""
    return np.sqrt(w - c) * np.sqrt(w + c)


def _deflection_sie(x, y, q):
    """Closed-form complex deflection alpha* = alpha_x - i*alpha_y of the
    SIE (theta_E = 1, major axis along x, intermediate-axis convention).

    :param x: x-coordinate (units of theta_E)
    :param y: y-coordinate (units of theta_E)
    :param q: axis ratio
    :return: complex deflection alpha*
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = x + 1j * y
    center = np.abs(z) == 0.0
    if np.any(center):  # direction undefined at the center; keep NaN-free
        z = np.where(center, 1e-12, z)
        x = np.where(center, 1e-12, x)
    zbar = np.conj(z)
    if q >= 1.0 - 1e-12:
        return zbar / np.abs(z)
    mz = np.sqrt(q * x ** 2 + y ** 2 / q)
    k = np.sqrt((1.0 - q ** 2) / q)
    rho = (1.0 - q) / (1.0 + q)
    s_b = _sqrt_branch(z, k * mz)
    w = k * mz / z
    asin_w = -1j * np.log(1j * w + s_b / z)  # branch-consistent arcsin
    return asin_w / k - mz / (z + s_b) + (zbar - rho * z) / (2.0 * mz)


def _lagrange_weights(grid, val):
    """4-point Lagrange cubic stencil (start index, weights) on a 1D grid."""
    val = float(np.clip(val, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, val) - 1, 0, len(grid) - 2))
    i0 = int(np.clip(j - 1, 0, len(grid) - 4))
    xs = grid[i0:i0 + 4]
    w = np.ones(4)
    for a in range(4):
        for b in range(4):
            if a != b:
                w[a] *= (val - xs[b]) / (xs[a] - xs[b])
    return i0, w


_INTERP_CACHE = {}


def _get_interp(table_path, cache_size=32):
    """Load each table once per process and share it between all profile
    instances (the npz decompress is expensive)."""
    key = os.path.abspath(table_path)
    if key not in _INTERP_CACHE:
        _INTERP_CACHE[key] = _GInterp(table_path, cache_size)
    return _INTERP_CACHE[key]


class _GInterp(object):
    """Interpolator for the tabulated ratio g = alpha*_model/alpha*_SIE."""

    def __init__(self, table_path, cache_size=32):
        t = np.load(table_path)
        self._logR = t["logR"]
        self._phi = t["phi"]  # [0, pi]; g is pi-periodic (point symmetry)
        self._pgrids = (t["q0"], t["dq"], t["dphi"], t["gamma"])
        self._g = (t["g_re"].astype(np.float64)
                   + 1j * t["g_im"].astype(np.float64))
        self._cache = {}
        self._cache_size = cache_size
        npad = 4
        d = self._phi[1] - self._phi[0]
        self._phi_pad = np.concatenate(
            [self._phi[0] + d * np.arange(-npad, 0), self._phi,
             self._phi[-1] + d * np.arange(1, npad + 1)])
        self._npad = npad

    def _slice_2d(self, q0, dq, dphi, gamma):
        key = (round(float(q0), 10), round(float(dq), 10),
               round(float(dphi), 10), round(float(gamma), 10))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        (i0, wq), (j0, wd), (k0, wp), (l0, wgam) = (
            _lagrange_weights(g, v)
            for g, v in zip(self._pgrids, (q0, dq, dphi, gamma)))
        block = self._g[:, :, i0:i0 + 4, j0:j0 + 4, k0:k0 + 4, l0:l0 + 4]
        g2d = np.einsum("rpijkl,i,j,k,l->rp", block, wq, wd, wp, wgam)
        n = self._npad
        g2d_pad = np.concatenate(
            [g2d[:, -n - 1:-1], g2d, g2d[:, 1:n + 1]], axis=1)
        spl = (RectBivariateSpline(self._logR, self._phi_pad,
                                   g2d_pad.real, kx=3, ky=3),
               RectBivariateSpline(self._logR, self._phi_pad,
                                   g2d_pad.imag, kx=3, ky=3))
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = spl
        return spl

    def alpha(self, x, y, q0, dq, dphi, gamma):
        """(alpha_x, alpha_y) in theta_E = 1 units, major axis along x."""
        R = np.hypot(x, y)
        logR = np.clip(np.log10(np.maximum(R, 1e-300)),
                       self._logR[0], self._logR[-1])
        phi = np.arctan2(y, x) % np.pi
        spl_re, spl_im = self._slice_2d(q0, dq, dphi, gamma)
        g = spl_re.ev(logR, phi) + 1j * spl_im.ev(logR, phi)
        a_star = g * _deflection_sie(x, y, q0)
        if gamma != 2.0:
            a_star = a_star * np.maximum(R, 1e-300) ** (2.0 - gamma)
        return a_star.real, -a_star.imag


class EPLQGrad(LensProfileBase):
    """Elliptical power law with log-linear ellipticity gradient and
    isodensity twist, parametrized with (q0, phi_G)."""

    profile_name = "EPL_QGRAD"
    param_names = ["theta_E", "gamma", "q0", "dq", "dphi", "phi_G",
                   "center_x", "center_y"]
    lower_limit_default = {"theta_E": 0.0, "gamma": 1.6, "q0": 0.3,
                           "dq": -0.2, "dphi": -0.2, "phi_G": -np.pi,
                           "center_x": -100, "center_y": -100}
    upper_limit_default = {"theta_E": 100.0, "gamma": 2.5, "q0": 1.0,
                           "dq": 0.2, "dphi": 0.2, "phi_G": np.pi,
                           "center_x": 100, "center_y": 100}

    def __init__(self, table_path=_DEFAULT_TABLE):
        """
        :param table_path: path to the precomputed npz table
        """
        self._interp = _get_interp(table_path)
        super().__init__()

    def derivatives(self, x, y, theta_E, gamma, q0, dq, dphi, phi_G,
                    center_x=0, center_y=0):
        """Deflection angles.

        :param x: x-coordinate [arcsec]
        :param y: y-coordinate [arcsec]
        :param theta_E: Einstein radius [arcsec]
        :param gamma: 3D logarithmic density slope (gamma = 2: isothermal)
        :param q0: axis ratio of the isodensity contour at theta_E
        :param dq: d(axis ratio) / d ln(r)
        :param dphi: d(position angle) / d ln(r) [radians per e-fold]
        :param phi_G: position angle at theta_E [radians]
        :param center_x: deflector x-center [arcsec]
        :param center_y: deflector y-center [arcsec]
        :return: alpha_x, alpha_y
        """
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        c, s = np.cos(phi_G), np.sin(phi_G)
        dx, dy = x - center_x, y - center_y
        u = (c * dx + s * dy) / theta_E
        v = (-s * dx + c * dy) / theta_E
        au, av = self._interp.alpha(u, v, q0, dq, dphi, gamma)
        au, av = theta_E * au, theta_E * av
        f_x, f_y = c * au - s * av, s * au + c * av
        if scalar:
            return float(f_x[0]), float(f_y[0])
        return f_x, f_y

    def function(self, x, y, theta_E, gamma, q0, dq, dphi, phi_G,
                 center_x=0, center_y=0):
        """Lensing potential via psi(x) = int_0^1 dt x . alpha(t x), with
        psi(center) = 0 (only potential differences are meaningful)."""
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        c, s = np.cos(phi_G), np.sin(phi_G)
        dx, dy = x - center_x, y - center_y
        u = (c * dx + s * dy) / theta_E
        v = (-s * dx + c * dy) / theta_E
        tg, wg = np.polynomial.legendre.leggauss(32)
        t = 0.5 * (tg + 1.0)
        w = 0.5 * wg
        psi = np.zeros_like(u)
        for ti, wi in zip(t, w):
            au, av = self._interp.alpha(ti * u, ti * v, q0, dq, dphi,
                                        gamma)
            psi += wi * (u * au + v * av)
        psi = theta_E ** 2 * psi
        return float(psi[0]) if scalar else psi

    def hessian(self, x, y, theta_E, gamma, q0, dq, dphi, phi_G,
                center_x=0, center_y=0, diff=1e-4):
        """Hessian by central differences of the deflection.

        NOTE: binds EPLQGrad.derivatives explicitly (not self.derivatives)
        so that subclasses with different parametrizations (e.g. e1/e2)
        can reuse this method after converting their parameters.
        """
        h = diff * theta_E
        args = (theta_E, gamma, q0, dq, dphi, phi_G, center_x, center_y)
        deriv = EPLQGrad.derivatives
        ax_px, ay_px = deriv(self, x + h, y, *args)
        ax_mx, ay_mx = deriv(self, x - h, y, *args)
        ax_py, ay_py = deriv(self, x, y + h, *args)
        ax_my, ay_my = deriv(self, x, y - h, *args)
        f_xx = (ax_px - ax_mx) / (2 * h)
        f_yy = (ay_py - ay_my) / (2 * h)
        f_xy = 0.5 * ((ax_py - ax_my) / (2 * h) + (ay_px - ay_mx) / (2 * h))
        return f_xx, f_xy, f_xy, f_yy


class EPLQGradEllipse(EPLQGrad):
    """Same profile parametrized with (e1, e2) in the standard lenstronomy
    ellipticity convention (recommended for sampling; avoids the
    position-angle wraparound degeneracy). The axis ratio recovered from
    (e1, e2) is q0, the axis ratio of the isodensity contour at theta_E."""

    profile_name = "EPL_QGRAD_ELL"
    param_names = ["theta_E", "gamma", "e1", "e2", "dq", "dphi",
                   "center_x", "center_y"]
    lower_limit_default = {"theta_E": 0.0, "gamma": 1.6, "e1": -0.5,
                           "e2": -0.5, "dq": -0.2, "dphi": -0.2,
                           "center_x": -100, "center_y": -100}
    upper_limit_default = {"theta_E": 100.0, "gamma": 2.5, "e1": 0.5,
                           "e2": 0.5, "dq": 0.2, "dphi": 0.2,
                           "center_x": 100, "center_y": 100}

    @staticmethod
    def _convert(e1, e2):
        from lenstronomy.Util.param_util import ellipticity2phi_q
        phi_G, q0 = ellipticity2phi_q(e1, e2)
        return q0, phi_G

    def derivatives(self, x, y, theta_E, gamma, e1, e2, dq, dphi,
                    center_x=0, center_y=0):
        q0, phi_G = self._convert(e1, e2)
        return super().derivatives(
            x, y, theta_E, gamma, q0, dq, dphi, phi_G, center_x, center_y)

    def function(self, x, y, theta_E, gamma, e1, e2, dq, dphi,
                 center_x=0, center_y=0):
        q0, phi_G = self._convert(e1, e2)
        return super().function(
            x, y, theta_E, gamma, q0, dq, dphi, phi_G, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, e1, e2, dq, dphi,
                center_x=0, center_y=0, diff=1e-4):
        q0, phi_G = self._convert(e1, e2)
        return super().hessian(
            x, y, theta_E, gamma, q0, dq, dphi, phi_G, center_x, center_y,
            diff=diff)


class EPL_QGRAD_MULTIPOLE_M1M3M4(LensProfileBase):
    """EPL_QGRAD mass profile (elliptical power law with ellipticity gradient
    and isodensity twist) combined with three elliptical multipole terms of
    order m=1, m=3 and m=4 (exact for general axis ratio q), concentric with
    the EPL.

    The multipole reference ellipse is the isodensity contour of the smooth
    profile at the Einstein radius: axis ratio q and orientation phi from
    (e1, e2). With dq = dphi = 0 this reproduces EPL_MULTIPOLE_M1M3M4_ELL
    exactly; with nonzero gradients/twists the multipole pattern remains
    anchored to the theta_E contour (mismatch away from the annulus is
    second order in the multipole amplitude).

    See also documentation of EPL_MULTIPOLE_M1M3M4_ELL,
    lenstronomy.LensModel.Profiles.epl_qgrad and
    lenstronomy.LensModel.Profiles.multipole for details.
    """

    profile_name = "EPL_QGRAD_MULTIPOLE_M1M3M4"
    param_names = [
        "theta_E",
        "gamma",
        "e1",
        "e2",
        "dq",
        "dphi",
        "center_x",
        "center_y",
        "a1_a",
        "delta_phi_m1",
        "a3_a",
        "delta_phi_m3",
        "a4_a",
        "delta_phi_m4",
    ]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.6,
        "e1": -0.5,
        "e2": -0.5,
        "dq": -0.2,
        "dphi": -0.2,
        "center_x": -100,
        "center_y": -100,
        "a1_a": -0.2,
        "delta_phi_m1": -np.pi,
        "a3_a": -0.2,
        "delta_phi_m3": -np.pi / 6,
        "a4_a": -0.2,
        "delta_phi_m4": -np.pi / 8,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "dq": 0.2,
        "dphi": 0.2,
        "center_x": 100,
        "center_y": 100,
        "a1_a": 0.2,
        "delta_phi_m1": np.pi,
        "a3_a": 0.2,
        "delta_phi_m3": np.pi / 6,
        "a4_a": 0.2,
        "delta_phi_m4": np.pi / 8,
    }

    def __init__(self, table_path=_DEFAULT_TABLE):
        from lenstronomy.LensModel.Profiles.multipole import (
            EllipticalMultipole)
        self._qgrad = EPLQGradEllipse(table_path)
        self._multipole = EllipticalMultipole()
        super(EPL_QGRAD_MULTIPOLE_M1M3M4, self).__init__()

    def _param_split(
            self,
            theta_E,
            gamma,
            e1,
            e2,
            dq,
            dphi,
            a1_a,
            delta_phi_m1,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=0,
            center_y=0,
    ):
        """This function splits the keyword arguments for the EPL_QGRAD and
        multipole profiles.

        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis); the
            recovered axis ratio is that of the isodensity contour at
            theta_E
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param dq: change in axis ratio per e-fold in radius
            (ellipticity gradient)
        :param dphi: change in position angle per e-fold in radius
            [radians] (isodensity twist)
        :param a1_a: amplitude of the m=1 mutipole perturbation from pure
            elliptical shape related to the physical amplitude of the
            MULTIPOLE_ELL profile by a scaling theta_E
        :param delta_phi_m1: orientation of the m=1 multipole perturbation
            relative to the semi-major axis of the EPL profile (NB: this is
            a value of eccentric anomaly, NOT a polar angle ! The
            corresponding polar angle depends on the axis ratio and
            orientation of the reference ellipse )
        :param a3_a: amplitude of the m=3 multiple deviation from pure
            elliptical shape related to the physical amplitude of the
            MULTIPOLE_ELL profile by a scaling theta_E
        :param delta_phi_m3: orientation of the m=3 multipole perturbation
            relative to the semi-major axis of the EPL profile (NB: this is
            a value of eccentric anomaly, NOT a polar angle ! The
            corresponding polar angle depends on the axis ratio and
            orientation of the reference ellipse )
        :param a4_a: amplitude of the m=4 multipole deviation from pure
            elliptical shape related to the physical amplitude of the
            MULTIPOLE_ELL profile by a scaling theta_E
        :param delta_phi_m4: orientation of the m=4 multipole perturbation
            relative to the semi-major axis of the EPL profile (NB: this is
            a value of eccentric anomaly, NOT a polar angle ! The
            corresponding polar angle depends on the axis ratio and
            orientation of the reference ellipse )
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: the keyword arguments for the joint profile
        """
        import lenstronomy.Util.param_util as param_util
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        kwargs_qgrad = {
            "theta_E": theta_E,
            "gamma": gamma,
            "e1": e1,
            "e2": e2,
            "dq": dq,
            "dphi": dphi,
            "center_x": center_x,
            "center_y": center_y,
        }
        kwargs_multipole_m1 = {
            "m": 1,
            "a_m": a1_a * theta_E,
            "varphi_m": delta_phi_m1,
            "q": q,
            "phi_ref": phi,
            "center_x": center_x,
            "center_y": center_y,
            "r_E": theta_E,
        }
        kwargs_multipole_m3 = {
            "m": 3,
            "a_m": a3_a * theta_E,
            "varphi_m": delta_phi_m3,
            "q": q,
            "phi_ref": phi,
            "center_x": center_x,
            "center_y": center_y,
            "r_E": theta_E,
        }
        kwargs_multipole_m4 = {
            "m": 4,
            "a_m": a4_a * theta_E,
            "varphi_m": delta_phi_m4,
            "q": q,
            "phi_ref": phi,
            "center_x": center_x,
            "center_y": center_y,
        }
        return (kwargs_qgrad, kwargs_multipole_m1, kwargs_multipole_m3,
                kwargs_multipole_m4)

    def function(
            self,
            x,
            y,
            theta_E,
            gamma,
            e1,
            e2,
            dq,
            dphi,
            a1_a,
            delta_phi_m1,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=0,
            center_y=0,
    ):
        """Computes the gravitational potential in units of theta_E^2.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :return: lensing potential (see _param_split for parameter
            documentation)
        """
        kwargs_qgrad, kwargs_multipole1, kwargs_multipole3, kwargs_multipole4 = (
            self._param_split(
                theta_E,
                gamma,
                e1,
                e2,
                dq,
                dphi,
                a1_a,
                delta_phi_m1,
                a3_a,
                delta_phi_m3,
                a4_a,
                delta_phi_m4,
                center_x=center_x,
                center_y=center_y,
            )
        )
        f_qgrad = self._qgrad.function(x, y, **kwargs_qgrad)
        f_multipole = self._multipole.function(x, y, **kwargs_multipole3)
        f_multipole += self._multipole.function(x, y, **kwargs_multipole4)
        f_multipole += self._multipole.function(x, y, **kwargs_multipole1)
        return f_qgrad + f_multipole

    def derivatives(
            self,
            x,
            y,
            theta_E,
            gamma,
            e1,
            e2,
            dq,
            dphi,
            a1_a,
            delta_phi_m1,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=0,
            center_y=0,
    ):
        """Computes the derivatives of the potential (deflection angles) in
        units of theta_E.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :return: alpha_x, alpha_y (see _param_split for parameter
            documentation)
        """
        kwargs_qgrad, kwargs_multipole1, kwargs_multipole3, kwargs_multipole4 = (
            self._param_split(
                theta_E,
                gamma,
                e1,
                e2,
                dq,
                dphi,
                a1_a,
                delta_phi_m1,
                a3_a,
                delta_phi_m3,
                a4_a,
                delta_phi_m4,
                center_x=center_x,
                center_y=center_y,
            )
        )
        f_x_qgrad, f_y_qgrad = self._qgrad.derivatives(x, y, **kwargs_qgrad)
        f_x_multipole3, f_y_multipole3 = self._multipole.derivatives(
            x, y, **kwargs_multipole3
        )
        f_x_multipole4, f_y_multipole4 = self._multipole.derivatives(
            x, y, **kwargs_multipole4
        )
        f_x_multipole1, f_y_multipole1 = self._multipole.derivatives(
            x, y, **kwargs_multipole1
        )
        f_x = f_x_qgrad + f_x_multipole3 + f_x_multipole4 + f_x_multipole1
        f_y = f_y_qgrad + f_y_multipole3 + f_y_multipole4 + f_y_multipole1
        return f_x, f_y

    def hessian(
            self,
            x,
            y,
            theta_E,
            gamma,
            e1,
            e2,
            dq,
            dphi,
            a1_a,
            delta_phi_m1,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=0,
            center_y=0,
    ):
        """Computes the components of the hessian matrix (second derivatives
        of the potential).

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :return: f_xx, f_xy, f_yx, f_yy (see _param_split for parameter
            documentation)
        """
        kwargs_qgrad, kwargs_multipole1, kwargs_multipole3, kwargs_multipole4 = (
            self._param_split(
                theta_E,
                gamma,
                e1,
                e2,
                dq,
                dphi,
                a1_a,
                delta_phi_m1,
                a3_a,
                delta_phi_m3,
                a4_a,
                delta_phi_m4,
                center_x=center_x,
                center_y=center_y,
            )
        )
        f_xx_qgrad, f_xy_qgrad, f_yx_qgrad, f_yy_qgrad = self._qgrad.hessian(
            x, y, **kwargs_qgrad
        )
        (
            f_xx_multipole3,
            f_xy_multipole3,
            f_yx_multipole3,
            f_yy_multipole3,
        ) = self._multipole.hessian(x, y, **kwargs_multipole3)
        (
            f_xx_multipole4,
            f_xy_multipole4,
            f_yx_multipole4,
            f_yy_multipole4,
        ) = self._multipole.hessian(x, y, **kwargs_multipole4)
        (
            f_xx_multipole1,
            f_xy_multipole1,
            f_yx_multipole1,
            f_yy_multipole1,
        ) = self._multipole.hessian(x, y, **kwargs_multipole1)
        f_xx = f_xx_qgrad + f_xx_multipole3 + f_xx_multipole4 + f_xx_multipole1
        f_xy = f_xy_qgrad + f_xy_multipole3 + f_xy_multipole4 + f_xy_multipole1
        f_yx = f_yx_qgrad + f_yx_multipole3 + f_yx_multipole4 + f_yx_multipole1
        f_yy = f_yy_qgrad + f_yy_multipole3 + f_yy_multipole4 + f_yy_multipole1
        return f_xx, f_xy, f_yx, f_yy