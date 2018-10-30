import numpy as np

class TBurk(object):

    """
    a cored Burkert profile, resembling a cored NFW profile

    relation are: R_200 = c * Rs

    """
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

    def density_2d(self, x, y, Rs, rho0, r_trunc, p, center_x=0, center_y=0):



        pass

    def mass_3d(self, Rs, rho0, r_trunc, p):
        pass

    def mass_2d(self, Rs, rho0, r_trunc, p):
        pass

    def _F(self, x, p, tau):

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
            inds_equal = np.where(x == p)
            x[inds_equal] += 1e-6
            inds_greater = np.where(x > p)
            inds_less = np.where(x < p)

            kappa = np.zeros_like(x)

            x_less = x[inds_less]
            x_greater = x[inds_greater]

            kappa[inds_less] = ((2 * (p ** 2 - tau ** 2) * np.arctan(p / np.sqrt(-p ** 2 + x_less ** 2))) /
                    (np.sqrt(-p ** 2 + x_less ** 2) * (p ** 2 + tau ** 2)) +
                    np.log((-p + np.sqrt(p ** 2 + x_less ** 2)) / (p + np.sqrt(p ** 2 + x_less ** 2))) / np.sqrt(p ** 2 + x_less ** 2) -
                    (np.pi * ((-(1 / np.sqrt(-p ** 2 + x_less ** 2)) + 1 / np.sqrt(p ** 2 + x_less ** 2)) * tau ** 2 +
                     p ** 2 * (1 / np.sqrt(-p ** 2 + x_less ** 2) + 1 / np.sqrt(p ** 2 + x_less ** 2) - 2 / np.sqrt(x_less ** 2 +
                        tau ** 2))) + (2 * p *tau * np.log((-tau + np.sqrt(x_less ** 2 + tau ** 2)) / (tau + np.sqrt(x_less ** 2 + tau ** 2)))) /
                        np.sqrt(x_less ** 2 + tau ** 2)) / (p ** 2 + tau ** 2)) / (2. * p * (p ** 2 - tau ** 2))

            kappa[inds_greater] = ((2 * (p ** 2 - tau ** 2) * np.arctan(p / np.sqrt(-p ** 2 + x_greater ** 2))) /
                                (np.sqrt(-p ** 2 + x_greater ** 2) * (p ** 2 + tau ** 2)) +
                                np.log((-p + np.sqrt(p ** 2 + x_greater ** 2)) / (
                                            p + np.sqrt(p ** 2 + x_greater ** 2))) / np.sqrt(p ** 2 + x_greater ** 2) -
                                (np.pi * ((-(1 / np.sqrt(-p ** 2 + x_greater ** 2)) + 1 / np.sqrt(
                                    p ** 2 + x_greater ** 2)) * tau ** 2 +
                                          p ** 2 * (1 / np.sqrt(-p ** 2 + x_greater ** 2) + 1 / np.sqrt(
                                            p ** 2 + x_greater ** 2) - 2 / np.sqrt(x_greater ** 2 +
                                                                                tau ** 2))) + (2 * p * tau * np.log(
                                    (-tau + np.sqrt(x_greater ** 2 + tau ** 2)) / (
                                                tau + np.sqrt(x_greater ** 2 + tau ** 2)))) /
                                 np.sqrt(x_greater ** 2 + tau ** 2)) / (p ** 2 + tau ** 2)) / (
                                           2. * p * (p ** 2 - tau ** 2))

        else:
            if x > p:

                kappa = ((np.np.pi*(-(tau**2*np.np.sqrt((p**2 - x**2)*(x**2 + tau**2))) +
                  p**2*(2*np.np.sqrt(p**4 - x**4) - np.np.sqrt((p**2 - x**2)*(x**2 + tau**2)))) -
               (p**2 - tau**2)*np.np.sqrt((p**2 + x**2)*(x**2 + tau**2))*
                np.np.log((p + np.np.sqrt(p**2 - x**2))/(p - np.np.sqrt(p**2 - x**2))) -
               (p**2 + tau**2)*np.np.sqrt((p**2 - x**2)*(x**2 + tau**2))*
                np.np.log((p + np.np.sqrt(p**2 + x**2))/(-p + np.np.sqrt(p**2 + x**2))) +
               2*p*np.np.sqrt(p**4 - x**4)*tau*np.np.log((-tau + np.np.sqrt(x**2 + tau**2))/(tau + np.np.sqrt(x**2 + tau**2)))
               ))/(2.*np.np.sqrt((p**4 - x**4)*(x**2 + tau**2))*(p**5 - p*tau**4))

            elif x < p:
                kappa = ((2 * (p ** 2 - tau ** 2) * np.arctan(p / np.sqrt(-p ** 2 + x ** 2))) /
                    (np.sqrt(-p ** 2 + x ** 2) * (p ** 2 + tau ** 2)) +
                    np.log((-p + np.sqrt(p ** 2 + x ** 2)) / (p + np.sqrt(p ** 2 + x ** 2))) / np.sqrt(p ** 2 + x ** 2) -
                    (np.pi * ((-(1 / np.sqrt(-p ** 2 + x ** 2)) + 1 / np.sqrt(p ** 2 + x ** 2)) * tau ** 2 +
                     p ** 2 * (1 / np.sqrt(-p ** 2 + x ** 2) + 1 / np.sqrt(p ** 2 + x ** 2) - 2 / np.sqrt(x ** 2 +
                        tau ** 2))) + (2 * p *tau * np.log((-tau + np.sqrt(x ** 2 + tau ** 2)) / (tau + np.sqrt(x ** 2 + tau ** 2)))) /
                        np.sqrt(x ** 2 + tau ** 2)) / (p ** 2 + tau ** 2)) / (2. * p * (p ** 2 - tau ** 2))

        return kappa

t = TBurk()
x = np.linspace(1, 1.4, 10000)
kappa1 = t._F(x, 0.2, 50)
kappa2 = t._F(x, 0.4, 1000)
import matplotlib.pyplot as plt
plt.loglog(x, kappa1)
plt.loglog(x, kappa2)
plt.show()


        

       

