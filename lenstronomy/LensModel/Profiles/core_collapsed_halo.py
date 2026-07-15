__author__ = "dgilman"

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.pseudo_double_powerlaw import PseudoDoublePowerlaw
__all__ = ["CoreCollapsedHalo"]


class CoreCollapsedHalo(LensProfileBase):

    profile_name = "CORE_COLLAPSED_HALO"
    param_names = ["center_x", "center_y", "Rs_inner", "Rs_outer", "alpha_Rs_inner", "alpha_Rs_outer",
                   "r_trunc", "gamma_inner", "gamma_outer"]
    lower_limit_default = {"Rs": 0, "alpha_Rs": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"Rs": 100, "alpha_Rs": 10, "center_x": 100, "center_y": 100}

    def __init__(self):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated
         (returning zeros outside)
        """
        self._profile_inner = PseudoDoublePowerlaw()
        self._profile_outer = TNFW()
        super(CoreCollapsedHalo, self).__init__()

    def _split_kwargs(self, center_x, center_y, Rs_inner, Rs_outer, alpha_Rs_inner, alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer):
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
        kwargs_inner, kwargs_outer = self._split_kwargs(center_x, center_y, Rs_inner, Rs_outer, alpha_Rs_inner,
                                                       alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer)
        f_x_inner, f_y_inner = self._profile_inner.derivatives(x, y, **kwargs_inner)
        f_x_outer, f_y_outer = self._profile_outer.derivatives(x, y, **kwargs_outer)
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
        kwargs_inner, kwargs_outer = self._split_kwargs(center_x, center_y, Rs_inner, Rs_outer, alpha_Rs_inner,
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
        kwargs_inner, kwargs_outer = self._split_kwargs(0.0, 0.0, Rs_inner, Rs_outer, alpha_Rs_inner,
                                                       alpha_Rs_outer, r_trunc, gamma_inner, gamma_outer)
        del kwargs_inner['center_x']
        del kwargs_inner['center_y']
        del kwargs_outer['center_x']
        del kwargs_outer['center_y']
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
