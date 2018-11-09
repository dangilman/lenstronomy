import numpy as np
from scipy.integrate import quad

class TBurk(object):

    """
    a cored Burkert profile, resembling a cored NFW profile.

    relation are: R_200 = c * Rs
    the parameter p is defined as p = r_core / rs

    """
    _dx = 1e-3

    param_names = ['Rs', 'theta_Rs', 'p', 'r_trunc', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'theta_Rs': 0, 'p':0, 'r_trunc': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'theta_Rs': 10, 'p':0.9, 'r_trunc': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, Rs, rho0, r_trunc, p, center_x=0, center_y=0):
        pass

    def derivatives(self, x, y, Rs, rho0, r_trunc, p, center_x=0, center_y=0):

        X = (x**2 + y**2) ** 0.5 * Rs ** -1
        if isinstance(X, np.ndarray):
            X[np.where(X < 0.001)] = 0.001
        else:
            X = min(0.001, X)

        tau = r_trunc / Rs

        x_ = x - center_x
        y_ = y - center_y

        mproj = p * self._projected_mass_integral(X, p, tau)

        a = 4 * rho0 * Rs * mproj * X ** -2

        return a * x_, a * y_

    def hessian(self, x, y, Rs, rho0, r_trunc, p, center_x=0, center_y=0):
        pass

    def density(self, R, Rs, rho0, r_trunc, p):

        y = R * Rs ** -1
        tau = r_trunc * Rs ** -1

        return p * rho0*tau**2*((p+y) * (p**2 + y**2) * (tau**2 + y**2)) ** -1

    def density_2d(self, x, y, Rs, rho0, r_trunc, r_core, center_x=0, center_y=0):

        x_ = x - center_x
        y_ = y - center_y

        X = (x_ **2 + y_ **2) ** 0.5 / Rs
        X[np.where(X < 0.001)] = 0.001

        p = r_core * Rs ** -1
        tau = r_trunc * Rs ** -1

        projection = p * self._projection_integral(X, p, tau)

        return Rs * rho0 * tau ** 2 * projection

    def mass_3d(self, Rs, rho0, r_trunc, r_core):
        pass

    def _projected_mass_integral(self, x, p, tau):

        if isinstance(x, list):
            x = np.array(x)

        m2d = self._projection_mass_numerical(x, p ,tau)

        return 2 * np.pi * m2d

    def _projection_mass_numerical(self, xmax, p, tau):

        if isinstance(xmax, int) or isinstance(xmax, float):
            projected_m = self._integrate_mprojected(xmax, p, tau)
        else:
            shape_x = xmax.shape
            xmax_long = xmax.ravel()
            projected_m = np.zeros_like(xmax_long)
            for i, xi in enumerate(xmax.ravel()):
                projected_m[i] = self._integrate_mprojected(xi, p, tau)
            projected_m.reshape(shape_x)

        return projected_m

    def _mproj_integrand(self, x, p, tau):

        return x * self._projection_integral(x, p, tau)

    def _integrate_mprojected(self, xmax_value, p, tau):


        if xmax_value > p + self._dx:
            projected_m = quad(self._mproj_integrand, self._dx, p - self._dx, args=(p, tau))[0]
            projected_m += quad(self._mproj_integrand, p + self._dx, xmax_value, args=(p, tau))[0]
        else:
            projected_m = quad(self._mproj_integrand, self._dx, xmax_value, args=(p, tau))[0]


        return projected_m


    def _projection_mass_plessx(self, x, p, tau):
        pass

    def _projection_mass_pgreaterx(self, x, p, tau):

        F1 = np.sqrt(p ** 2 - x ** 2)
        F2 = np.sqrt(x ** 2 + tau ** 2)
        F3 = np.sqrt(p ** 2 + x ** 2)
        L1 = np.log(-1 + (2*p*(p + F1)) * x ** -2)
        L2 = np.log(1 + (2*p*(p + F3)) * x ** -2)
        L3 = np.log(1 + (2*tau*(tau + F2)) * x ** -2)

        numerator = np.pi * (
                    F1 * F2 * F3 * L2 * p ** 2 - F2 * L1 * p ** 4 + F2 * L1 * p ** 2 * x ** 2 - 2 * F1 * L3 * p * x ** 2 * tau +
                    F1 * F2 * F3 * L2 * tau ** 2 + F2 * L1 * p ** 2 * tau ** 2 - F2 * L1 * x ** 2 * tau ** 2 - 2 * F1 * L3 * p * tau ** 3 +
                    4 * F1 * F2 * p * tau ** 2 * np.log(4 * p) + 4 * F1 * F2 * p * tau ** 2 * np.log(
                x) + 2 * F1 * F2 * p * tau ** 2 * np.log(F2 - tau) -
                    4 * F1 * F2 * p * tau ** 2 * np.log(tau) + 2 * F1 * F2 * p * tau ** 2 * np.log(F2 - tau))

        denom = ((F1 * F2 * (p ** 5 - p * tau ** 4))) ** -1

        return numerator * denom

    def _projection_integral(self, x, p, tau):

        """
        Analytic solution of the projection integral
        :param x: 
        :param p: 
        :param tau: 
        :return: 
        """

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(x, np.ndarray):

            x[np.where(x == p)] = p + self._dx

            xlessp, xgreaterp, (low_inds, high_inds) = self._x_array_split(x, p)

            proj = np.zeros_like(x)
            proj[high_inds] = self._projection_plessx(xgreaterp, p, tau)
            proj[low_inds] = self._projection_pgreaterx(xlessp, p, tau)

        else:

            if x == p:
                x += self._dx

            if p < x:
                proj = self._projection_plessx(x, p, tau)
            else:
                proj = self._projection_pgreaterx(x, p, tau)

        return proj

    def _projection_plessx(self, x, p, tau):

        return 0.5*(np.pi*((1/np.sqrt(-p**2 + x**2) - 1/np.sqrt(p**2 + x**2))/(p**3 - p*tau**2) +
              (2*p*(-(1/np.sqrt(-p**2 + x**2)) + 1/np.sqrt(x**2 + tau**2)))/(p**4 - tau**4)) +
           (2*(p**2 - tau**2)*np.arctan(p/np.sqrt(-p**2 + x**2)))/
            (np.sqrt(-p**2 + x**2)*(p**2 + tau**2)*(p**3 - p*tau**2)) -
           (2*np.arctanh(p/np.sqrt(p**2 + x**2)))/(np.sqrt(p**2 + x**2)*(p**3 - p*tau**2)) +
           (4*tau*np.arctanh(tau/np.sqrt(x**2 + tau**2)))/(np.sqrt(x**2 + tau**2)*(p**4 - tau**4)))

    def _projection_pgreaterx(self, x, p, tau):

        return (np.pi * (-(np.sqrt(p ** 4 - x ** 4) / (p * (p ** 2 + x ** 2))) +
              (2 * p * np.sqrt((p ** 2 - x ** 2) * (x ** 2 + tau ** 2))) / (
                           (p ** 2 + tau ** 2) * (x ** 2 + tau ** 2)))) / (2. * np.sqrt(p ** 2 - x ** 2) * (p ** 2 - tau ** 2)) + ((2 * (-p ** 2 + tau ** 2) * np.arctanh(np.sqrt(p ** 2 - x ** 2) / p)) / (p * (p ** 2 + tau ** 2)) -
        (2 * np.sqrt(p ** 4 - x ** 4) * np.arctanh(p / np.sqrt(p ** 2 + x ** 2))) / (p * (p ** 2 + x ** 2)) -
         (2 *tau * np.sqrt((p ** 2 - x ** 2) * (x ** 2 + tau ** 2)) *
        np.log((x ** 2 + 2 *tau * (tau - np.sqrt(x ** 2 + tau ** 2))) / x ** 2)) /
        ((p ** 2 + tau ** 2) * (x ** 2 + tau ** 2))) / (
                    2. * np.sqrt(p ** 2 - x ** 2) * (p ** 2 - tau ** 2))

    def _x_array_split(self, x, ref):

        inds_lower = np.where(x < ref)

        inds_greater = np.where(x > ref)

        return x[inds_lower], x[inds_greater], (inds_lower, inds_greater)

from lenstronomy.LensModel.Profiles.tnfw import TNFW
tnfw = TNFW()
rho = 10000
Rs = 0.6
tau = 10
p = 0.5
x = np.linspace(0.1 * Rs, 5 * Rs, 100)

r_trunc = tau*Rs
r_core = p*Rs

t = TBurk()
import matplotlib.pyplot as plt
#print(np.log((x ** 2 + 2 *tau * (tau - np.sqrt(x ** 2 + tau ** 2))) / x ** 2))
dx, dy = t.derivatives(x, 0, Rs, 1, r_trunc, p)
dxtnfw, _ = tnfw.derivatives(x, 0 , Rs, 1, r_trunc)

#plt.loglog(x/Rs, t._projected_mass_integral(x, p, tau), color='k')
#plt.loglog(x/Rs, proj2, color='r')
plt.loglog(x/Rs, dx, color='r')
plt.loglog(x/Rs, dxtnfw, color='k')

plt.show()
