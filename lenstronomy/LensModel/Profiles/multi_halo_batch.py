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
        Vectorize deflection angle calls for identical halo classes at one lens plane. This will only work
        if the lens profile is broadcast-safe.
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param profile_name: lens profile name ("TNFW", "POINT_MASS", etc)
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

        # if try_broadcast:
        #     try:
        #         xr = x.reshape(1, -1); yr = y.reshape(1, -1)
        #         kw = {}
        #         for k in keys:
        #             col = np.array([d[k] for d in kwargs_list], float)
        #             # constant-across-halos params stay scalar (avoids tripping
        #             # scalar guards like `if n == 3`); only varying params broadcast
        #             kw[k] = col[:, None] if np.ptp(col) != 0 else float(col.flat[0])
        #         fx, fy = base.derivatives(xr, yr, **kw)   # -> (N, M)
        #         return np.asarray(fx).sum(0).reshape(x.shape), np.asarray(fy).sum(0).reshape(x.shape)
        #     except Exception:
        #         pass  # fall back
        # ax = np.zeros_like(x); ay = np.zeros_like(y)
        # for d in kwargs_list:
        #     fx, fy = base.derivatives(x, y, **d)
        #     ax = ax + fx; ay = ay + fy
        # return ax, ay
