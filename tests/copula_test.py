import os
import unittest
import numpy as np
import pandas as pd
import sys; sys.path.insert(0, "..")
from maxlike import copula
save = False


def normalize(u):
    return u / u.sum()


class Test(unittest.TestCase):
    
    @staticmethod
    def _path(test_name):
        data_folder = os.path.dirname(os.path.abspath(__file__))
        return "{}/data/copula/{}.csv".format(data_folder, test_name)

    def setUp(self):
        f_xy = pd.read_csv(self._path("gaussian")).values
        self.f_x = normalize(f_xy.sum(1))
        self.f_y = normalize(f_xy.sum(0))

    def check_is_copula(self, f_xy):
        self.assertTrue(f_xy.max() <= 1)
        self.assertTrue(f_xy.min() >= 0)
        np.testing.assert_allclose(f_xy.sum(), 1.0)
        np.testing.assert_allclose(f_xy.sum(0), self.f_y)
        np.testing.assert_allclose(f_xy.sum(1), self.f_x)

    def test_gaussian(self):
        indep_xy = self.f_x[:, None] * self.f_y[None, :]
        f_xy = copula.Gaussian(rho=0)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)
        f_xy = copula.Gaussian(rho=0.5)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        if save: pd.DataFrame(f_xy).to_csv(self._path("gaussian"), index=False)
        xy_static = pd.read_csv(self._path("gaussian")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_clayton(self):
        f_xy = copula.Clayton(rho=1.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        if save: pd.DataFrame(f_xy).to_csv(self._path("clayton"), index=False)
        xy_static = pd.read_csv(self._path("clayton")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_gumbel(self):
        indep_xy = self.f_x[:, None] * self.f_y[None, :]
        f_xy = copula.Gumbel(rho=1)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)
        
        f_xy = copula.Gumbel(rho=2.0)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        if save: pd.DataFrame(f_xy).to_csv(self._path("gumbel"), index=False)
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
        if save: pd.DataFrame(f_xy).to_csv(self._path("frank2"), index=False)
        xy_static = pd.read_csv(self._path("frank2")).values
        np.testing.assert_allclose(f_xy, xy_static)

    def test_amh(self):
        indep_xy = self.f_x[:, None] * self.f_y[None, :]
        f_xy = copula.AkiMikhailHaq(rho=0)([self.f_x, self.f_y]).values
        np.testing.assert_allclose(f_xy, indep_xy)

        f_xy = copula.AkiMikhailHaq(rho=0.5)([self.f_x, self.f_y]).values
        self.check_is_copula(f_xy)
        if save: pd.DataFrame(f_xy).to_csv(self._path("amh"), index=False)
        xy_static = pd.read_csv(self._path("amh")).values
        np.testing.assert_allclose(f_xy, xy_static)


if __name__ == "__main__":
    unittest.main(verbosity=1)
