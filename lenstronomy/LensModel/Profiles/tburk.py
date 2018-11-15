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

        mproj = self._projected_mass_integral(X, p, tau)

        a = 4 * rho0 * Rs * tau**2 * mproj * X ** -2

        return a * x_, a * y_

    def hessian(self, x, y, Rs, rho0, r_trunc, p, center_x=0, center_y=0):
        pass

    def density(self, R, Rs, rho0, r_trunc, p):

        y = R * Rs ** -1
        tau = r_trunc * Rs ** -1

        return rho0*tau**2*((p+y) * (p**2 + y**2) * (tau**2 + y**2)) ** -1

    def density_2d(self, x, y, Rs, rho0, r_trunc, r_core, center_x=0, center_y=0):

        x_ = x - center_x
        y_ = y - center_y

        X = (x_ **2 + y_ **2) ** 0.5 / Rs
        X[np.where(X < 0.001)] = 0.001

        p = r_core * Rs ** -1
        tau = r_trunc * Rs ** -1

        projection = self._projection_integral(X, p, tau)

        return Rs * rho0 * tau ** 2 * projection

    def mass_3d(self, Rs, rho0, r_trunc, r_core):
        pass

    def _projected_mass_integral(self, x, p, tau):

        x = np.array(x)

        x_shape_0 = x.shape

        integral = []

        for i, xi in enumerate(x.ravel()):

            mproj = self._integrate_mproj(xi, p, tau)
            integral.append(mproj)

        integral = np.array(integral)

        return 2 * np.pi * integral.reshape(x_shape_0)

    def _integrate_mproj(self, x, p, tau, x_min = 0.001, dx_min = 0.001):

        def _integrand(x,p,tau):
            kappa = self._projection_integral(x, p, tau)
            return x * kappa

        if x < p:

            integral = quad(_integrand, x_min, min(p-dx_min,x), args=(p, tau))[0]

        else:
            #print(x, p)
            integral_low = quad(_integrand, x_min, p - dx_min, args=(p, tau))[0]
            #print(integral_low)
            integral_high = quad(_integrand, p+dx_min, x, args=(p, tau))[0]
            #print(integral_high)
            integral = integral_low + integral_high
        return integral

    """
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
    """

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

    def _projection_pgreaterx(self, x, p, tau, d_arg = 0.005):

        def log_function(x,p,t):
            arg = (1 + 2*t*(t - np.sqrt(x**2+t**2)) * x ** -2)
            if isinstance(arg, np.ndarray):
                arg[np.where(arg<d_arg)] = d_arg
            else:
                arg = max(arg, d_arg)
            return np.log(arg)

        return (np.pi * (-(np.sqrt(p ** 4 - x ** 4) / (p * (p ** 2 + x ** 2))) +
              (2 * p * np.sqrt((p ** 2 - x ** 2) * (x ** 2 + tau ** 2))) / (
                           (p ** 2 + tau ** 2) * (x ** 2 + tau ** 2)))) / (2. * np.sqrt(p ** 2 - x ** 2) * (p ** 2 - tau ** 2)) + ((2 * (-p ** 2 + tau ** 2) * np.arctanh(np.sqrt(p ** 2 - x ** 2) / p)) / (p * (p ** 2 + tau ** 2)) -
        (2 * np.sqrt(p ** 4 - x ** 4) * np.arctanh(p / np.sqrt(p ** 2 + x ** 2))) / (p * (p ** 2 + x ** 2)) -
         (2 *tau * np.sqrt((p ** 2 - x ** 2) * (x ** 2 + tau ** 2)) *
          log_function(x, p, tau)) /
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
tau = 20
p = 0.4
x = np.linspace(0.1 * Rs, 50 * Rs, 1000)

r_trunc = tau*Rs
r_core = p*Rs

t = TBurk()

#print(t._integrate_mprojected(0.3, p, tau))
#a=input('continue')
import matplotlib.pyplot as plt
#print(np.log((x ** 2 + 2 *tau * (tau - np.sqrt(x ** 2 + tau ** 2))) / x ** 2))
dx, dy = t.derivatives(x, 0, Rs, 1, r_trunc, p)
dxtnfw, _ = tnfw.derivatives(x, 0 , Rs, 1, r_trunc)
norm = dxtnfw[-1] * dx[-1] ** -1
#mproj = t._projected_mass_integral(x, p, tau)

#plt.loglog(x/Rs, t._projected_mass_integral(x, p, tau), color='k')
#plt.loglog(x/Rs, proj2, color='r')
plt.plot(x/Rs, dx * norm, color='r')
plt.loglog(x/Rs, dxtnfw, color='k')

plt.show()
