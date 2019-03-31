import os
import unittest
import numpy as np
import pandas as pd
from maxlike import copula


def normalize(u):
    return u / u.sum()


data_folder = os.path.dirname(os.path.abspath(__file__)) + "/data/copula"


class Test(unittest.TestCase):

    f_x = normalize(np.array([0.00, 1.22, 1.63, 2.21, 1.11, 0.00, 0.00]))
    f_y = normalize(np.array([0.01, 1.95, 1.33, 3.45, 2.59, 0.33, 0.00]))
    
    @staticmethod
    def _path(test_name):
        return "{}/{}.csv".format(data_folder, test_name)

    def check_is_copula(self, f_xy):
        self.assertTrue(f_xy.max() <= 1)
        self.assertTrue(f_xy.min() >= 0)
        np.testing.assert_allclose(f_xy.sum(), 1.0)
        np.testing.assert_allclose(f_xy.sum(0), self.f_y)
        np.testing.assert_allclose(f_xy.sum(1), self.f_x)

    def test_gaussian(self):
        indep_xy = self.f_x[:, None] * self.f_y[None]
        f_xy = copula.Gaussian(rho=0)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)
        f_xy = copula.Gaussian(rho=0.5)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        xy_static = pd.read_csv(self._path("gaussian")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_clayton(self):
        f_xy = copula.Clayton(rho=1.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        xy_static = pd.read_csv(self._path("clayton")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_gumbel(self):
        indep_xy = self.f_x[:, None] * self.f_y[None]
        f_xy = copula.Gumbel(rho=1)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)
        
        f_xy = copula.Gumbel(rho=2.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        xy_static = pd.read_csv(self._path("gumbel")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_frank(self):
        f_xy = copula.Frank(rho=1.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        if save: pd.DataFrame(f_xy).to_csv(self._path("frank"), index=False)
        xy_static = pd.read_csv(self._path("frank")).values
        np.testing.assert_allclose(f_xy, xy_static)

        # testing for negative values
        f_xy = copula.Frank(rho=-1.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        xy_static = pd.read_csv(self._path("frank2")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_amh(self):
        indep_xy = self.f_x[:, None] * self.f_y[None]
        f_xy = copula.AkiMikhailHaq(rho=0)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)

        f_xy = copula.AkiMikhailHaq(rho=0.5)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        xy_static = pd.read_csv(self._path("amh")).values
        np.testing.assert_allclose(f_xy, xy_static)


if __name__ == "__main__":
    unittest.main(verbosity=1)
