"""
Generic batched wrapper: represents MANY halos of a SINGLE base profile living
on one lens plane as one lenstronomy profile. Written once, works for any base
profile. Automatically vectorizes across halos when the base profile is
broadcast-safe; otherwise falls back to an internal loop (still removes the
per-profile multiplane overhead).
"""
import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.profile_list_base import lens_class

def _instance(profile_name):
    # reuse lenstronomy's own registry so ANY registered profile works
    return lens_class(profile_name)

class MultiHaloBatch(LensProfileBase):

    """
    Represent many halos at a single lens plane as one lenstronomy profile; currently only implemented for
    ray tracing with the derivatives class method. Does not currently work with MCMC sampling
    """
    profile_name = "MULTI_HALO_BATCH"
    param_names = ["profile_name", "kwargs_list"]

    def __init__(self):
        self._cache = {}
        super().__init__()

    def _base(self, name):
        if name not in self._cache:
            self._cache[name] = _instance(name)
        return self._cache[name]

    def derivatives(self, x, y, profile_name, kwargs_list):
        """
        Vectorize deflection angle calls for identical lens profile classes at one lens plane. This will only work
        if the lens profile is broadcast-safe.
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param profile_name: lens profile name
        :param kwargs_list: keyword arguments corresponding to each profile
        :return: deflection angles at coordinates (x, y)
        """
        base = self._base(profile_name)
        x = np.asarray(x, float); y = np.asarray(y, float)
        keys = list(kwargs_list[0].keys()); N = len(kwargs_list)
        xr = x.reshape(1, -1);
        yr = y.reshape(1, -1)
        kw = {}
        for k in keys:
            col = np.array([d[k] for d in kwargs_list], float)
            # constant-across-halos params stay scalar (avoids tripping
            # scalar guards like `if n == 3`); only varying params broadcast
            kw[k] = col[:, None] if np.ptp(col) != 0 else float(col.flat[0])
        fx, fy = base.derivatives(xr, yr, **kw)  # -> (N, M)
        return np.asarray(fx).sum(0).reshape(x.shape), np.asarray(fy).sum(0).reshape(x.shape)

    def hessian(self, x, y, profile_name, kwargs_list, diff=1e-6):
        """
        Hessian (second derivatives) of the summed lens profiles at one lens plane, by central
        finite differences of the batched deflection. This works for ANY base profile whose
        derivatives() is broadcast-safe

        :param x: angular position (arcsec)
        :param y: angular position (arcsec)
        :param profile_name: base lens profile name
        :param kwargs_list: keyword arguments for each halo
        :param diff: finite-difference step (arcsec)
        :return: f_xx, f_xy, f_yx, f_yy summed over all halos at (x, y)
        """
        x = np.asarray(x, float);
        y = np.asarray(y, float)
        ax_xp, ay_xp = self.derivatives(x + diff, y, profile_name, kwargs_list)
        ax_xm, ay_xm = self.derivatives(x - diff, y, profile_name, kwargs_list)
        ax_yp, ay_yp = self.derivatives(x, y + diff, profile_name, kwargs_list)
        ax_ym, ay_ym = self.derivatives(x, y - diff, profile_name, kwargs_list)
        f_xx = (ax_xp - ax_xm) / (2 * diff)
        f_yy = (ay_yp - ay_ym) / (2 * diff)
        f_xy = (ax_yp - ax_ym) / (2 * diff)
        f_yx = (ay_xp - ay_xm) / (2 * diff)
        return f_xx, f_xy, f_yx, f_yy