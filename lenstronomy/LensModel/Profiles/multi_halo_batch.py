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
    Represent many halos at a single lens plane as one lenstronomy profile.
    Does not currently work with MCMC sampling.
    """
    profile_name = "MULTI_HALO_BATCH"
    param_names = ["profile_name", "kwargs_list"]
    max_elements = 2000000 # limit the size of the arrays to avoid memory spikes

    def __init__(self):
        self._cache = {}
        super().__init__()

    def _base(self, name):
        if name not in self._cache:
            self._cache[name] = _instance(name)
        return self._cache[name]

    def function(self, x, y, profile_name, kwargs_list):
        """
        Vectorize potential calls for identical lens profile classes at one
        lens plane. This will only work if the lens profile is broadcast-safe.
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param profile_name: lens profile name
        :param kwargs_list: keyword arguments corresponding to each profile
        :return: potential at coordinates (x, y)
        """
        base = self._base(profile_name)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shape = x.shape
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        n_points = x_flat.size

        keys = list(kwargs_list[0].keys())
        n_halos = len(kwargs_list)
        kw = {}
        for k in keys:
            col = np.array([d[k] for d in kwargs_list], float)
            # constant-across-halos params stay scalar (avoids tripping
            # scalar guards like `if n == 3`); only varying params broadcast
            kw[k] = col[:, None] if np.ptp(col) != 0 else float(col.flat[0])
        if n_halos > 1 and not any(np.ndim(v) for v in kw.values()):
            # every parameter is identical across halos, so nothing carries the halo
            # axis and broadcasting would collapse N halos into one; force one key
            # to keep the axis explicit
            kw[keys[0]] = np.full((n_halos, 1), kw[keys[0]], dtype=float)
        block = max(1, int(self.max_elements // max(n_halos, 1)))
        if block >= n_points:
            # small enough to do in one shot -- identical to the original path
            pot = base.function(
                x_flat.reshape(1, -1), y_flat.reshape(1, -1), **kw
            )
            return np.asarray(pot).sum(0).reshape(shape)

        pot_out = np.empty(n_points)
        for start in range(0, n_points, block):
            stop = min(start + block, n_points)
            pot = base.function(
                x_flat[start:stop].reshape(1, -1),
                y_flat[start:stop].reshape(1, -1),
                **kw
            )
            pot_out[start:stop] = np.asarray(pot).sum(0)
        return pot_out.reshape(shape)

    def derivatives(self, x, y, profile_name, kwargs_list):
        """Vectorize deflection angle calls for identical lens profile classes at one
        lens plane. This will only work if the lens profile is broadcast-safe.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param profile_name: lens profile name
        :param kwargs_list: keyword arguments corresponding to each profile
        :return: deflection angles at coordinates (x, y)
        """
        base = self._base(profile_name)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shape = x.shape
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        n_points = x_flat.size

        keys = list(kwargs_list[0].keys())
        n_halos = len(kwargs_list)
        kw = {}
        for k in keys:
            col = np.array([d[k] for d in kwargs_list], float)
            # constant-across-halos params stay scalar (avoids tripping
            # scalar guards like `if n == 3`); only varying params broadcast
            kw[k] = col[:, None] if np.ptp(col) != 0 else float(col.flat[0])
        if n_halos > 1 and not any(np.ndim(v) for v in kw.values()):
            # every parameter is identical across halos, so nothing carries the halo
            # axis and broadcasting would collapse N halos into one; force one key
            # to keep the axis explicit
            kw[keys[0]] = np.full((n_halos, 1), kw[keys[0]], dtype=float)
        block = max(1, int(self.max_elements // max(n_halos, 1)))
        if block >= n_points:
            # small enough to do in one shot -- identical to the original path
            fx, fy = base.derivatives(
                x_flat.reshape(1, -1), y_flat.reshape(1, -1), **kw
            )
            return (
                np.asarray(fx).sum(0).reshape(shape),
                np.asarray(fy).sum(0).reshape(shape),
            )

        out_x = np.empty(n_points)
        out_y = np.empty(n_points)
        for start in range(0, n_points, block):
            stop = min(start + block, n_points)
            fx, fy = base.derivatives(
                x_flat[start:stop].reshape(1, -1),
                y_flat[start:stop].reshape(1, -1),
                **kw
            )
            out_x[start:stop] = np.asarray(fx).sum(0)
            out_y[start:stop] = np.asarray(fy).sum(0)
        return out_x.reshape(shape), out_y.reshape(shape)

    def hessian(self, x, y, profile_name, kwargs_list, diff=1e-6):
        """
        Hessian (second derivatives) of the summed lens profiles at one lens plane, by central
        finite differences of the batched deflection

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