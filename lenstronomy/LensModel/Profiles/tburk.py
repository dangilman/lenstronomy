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

    def _g(self, x, p, tau):

        """
        solution of the projection integral
        :param x:
        :param p:
        :param tau:
        :return:
        """

        t2 = tau ** 2
        p2 = p ** 2
        x2 = x ** 2

        f1 = x2 - p2
        f2 = x2 + p2
        f3 = x2 + t2
        f4 = x2 - t2
        f5 = t2 + p2

        first_term = np.pi * (2*p * ((p2 ** 2 - t2 **2) * f3 ** 0.5) ** -1 -
                              (p * f5 * f1 ** 0.5) ** -1 -
                              (p*(p2 - t2)*f2**0.5)**-1)
        second_term = None
        

