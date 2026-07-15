__author__ = "dgilman"

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.point_mass import PointMass
__all__ = ["CoreCollapsedHaloBH"]


class CoreCollapsedHaloBH(LensProfileBase):

    profile_name = "CORE_COLLAPSED_HALO_BH"
    param_names = ["center_x", "center_y", "theta_E_inner", "Rs_outer", "alpha_Rs_outer", "r_trunc"]
    lower_limit_default = {"Rs": 0, "alpha_Rs": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"Rs": 100, "alpha_Rs": 10, "center_x": 100, "center_y": 100}

    def __init__(self):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated
         (returning zeros outside)
        """
        self._profile_inner = PointMass()
        self._profile_outer = TNFW()
        super(CoreCollapsedHaloBH, self).__init__()

    def _split_kwargs(self, center_x, center_y, theta_E_inner, Rs_outer, alpha_Rs_outer, r_trunc):
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
        kwargs_inner = {'theta_E': theta_E_inner, 'center_x': center_x, 'center_y': center_y}
        kwargs_outer = {'alpha_Rs': alpha_Rs_outer, 'center_x': center_x, 'center_y': center_y, 'Rs': Rs_outer,
                        'r_trunc': r_trunc}
        return kwargs_inner, kwargs_outer

    def derivatives(self, x, y, Rs_outer, theta_E_inner, alpha_Rs_outer, r_trunc, center_x=0, center_y=0):
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
        kwargs_inner, kwargs_outer = self._split_kwargs(center_x, center_y, theta_E_inner, Rs_outer,
                                                       alpha_Rs_outer, r_trunc)
        f_x_inner, f_y_inner = self._profile_inner.derivatives(x, y, **kwargs_inner)
        f_x_outer, f_y_outer = self._profile_outer.derivatives(x, y, **kwargs_outer)
        return f_x_inner + f_x_outer, f_y_inner + f_y_outer

    def hessian(self, x, y, Rs_outer, theta_E_inner, alpha_Rs_outer, r_trunc,
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
        kwargs_inner, kwargs_outer = self._split_kwargs(center_x, center_y, theta_E_inner, Rs_outer,
                                                        alpha_Rs_outer, r_trunc)
        f_xx_inner, f_xy_inner, f_yx_inner, f_yy_inner = self._profile_inner.hessian(x, y, **kwargs_inner)
        f_xx_outer, f_xy_outer, f_yx_outer, f_yy_outer = self._profile_outer.hessian(x, y, **kwargs_outer)
        return f_xx_inner + f_xx_outer, f_xy_inner + f_xy_outer, f_yx_inner + f_yx_outer, f_yy_inner + f_yy_outer
