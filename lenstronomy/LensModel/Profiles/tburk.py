import numpy as np

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
        pass

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

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(x, np.ndarray):

            x[np.where(x == p)] = p + self._dx
            xlow, xhigh, (low_inds, high_inds) = self._x_array_split(x, p)
            m2d = np.zeros_like(x)
            m2d[low_inds] = self._projection_mass_plessx(xlow, p, tau)
            m2d[high_inds] = self._projection_mass_pgreaterx(xhigh, p, tau)

        else:
            if x == p:
                x += self._dx

            if p < x:
                m2d = self._projection_mass_plessx(x, p, tau)
            else:
                m2d = self._projection_mass_pgreaterx(x, p, tau)

        return 2 * np.pi * m2d

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

        return (0.5*(np.pi*(-(np.sqrt(p**4 - x**4)/(p*(p**2 + x**2))) + 
              (2*p*np.sqrt((p**2 - x**2)*(x**2 + tau**2)))/((p**2 + tau**2)*(x**2 + tau**2))) +
           (2*p*np.arctanh(np.sqrt(p**2 - x**2)/p))/(p**2 + x**2) +
           (2*x**2*np.arctanh(np.sqrt(p**2 - x**2)/p))/(p*(p**2 + x**2)) -
           (4*p*x**2*np.arctanh(np.sqrt(p**2 - x**2)/p))/((p**2 + tau**2)*(x**2 + tau**2)) -
           (4*p*tau**2*np.arctanh(np.sqrt(p**2 - x**2)/p))/((p**2 + tau**2)*(x**2 + tau**2)) -
           (2*np.sqrt(p**4 - x**4)*np.arctanh(p/np.sqrt(p**2 + x**2)))/(p*(p**2 + x**2)) -
           (2*tau*np.sqrt((p**2 - x**2)*(x**2 + tau**2))*
              np.log((x**2 + 2*tau*(tau - np.sqrt(x**2 + tau**2)))/x**2))/
            ((p**2 + tau**2)*(x**2 + tau**2))))/(np.sqrt(p**2 - x**2)*(p**2 - tau**2))

    def _x_array_split(self, x, ref):
        inds_lower = np.where(x < ref)

        inds_greater = np.where(x > ref)

        return x[inds_lower], x[inds_greater], (inds_lower, inds_greater)


from lenstronomy.LensModel.Profiles.tnfw import TNFW
tnfw = TNFW()
rho = 10000
Rs = 0.6
tau = 2
p = .04
x = np.logspace(np.log10(0.001*Rs), np.log10(5*Rs), 100)
print(x)
t = TBurk()
kappa1 = p * t.density_2d(x, 0, Rs, rho, tau * Rs, p * Rs)
kappa2 = tnfw.density_2d(x, 0, Rs, rho, tau * Rs)
import matplotlib.pyplot as plt
plt.loglog(x, kappa1, color='r')
plt.loglog(x, kappa2, color = 'k')
plt.show()


        

       

