"""
Checks the analytical first and second derivatives of every Func (and of
the composite models used in the other tests) against finite differences,
through both foo.eval (used by MaxLike.fit) and foo.grad / foo.hess.
"""
import unittest
import numpy as np
from maxlike.func import (
    X, Take, Scalar, Vector, Constant, Linear, Quadratic, Exp, Log, Sigmoid,
    Logistic, Poisson, LogNegativeBinomial, NegativeBinomial,
    Sum, Product, Collapse, MarkovMatrix, MarkovVector, CollapseMatrix)
from .numdiff import derivative_errors


rng = np.random.RandomState(0)
U = rng.uniform(.5, 1.5, 3)
V = rng.uniform(.5, 1.5, 3)
A = rng.normal(0, .2, 3)
B = rng.normal(0, .2, 3)
H = .3


def attack_defence():
    # y[i, j, k] = a[i] - b[j] + h * (k - 1/2)
    f = Sum(3)
    f.add(X(), 0, 0)
    f.add(-X(), 1, 1)
    f.add(Vector(np.arange(2) - .5), 2, 2)
    return f


def same_param_two_axes():
    # y[i, j] = a[i] - a[j]: one parameter mapped to two feature axes
    f = Sum(2)
    f.add(X(), 0, 0)
    f.add(-X(), 0, 1)
    return f


def home_away(swap):
    f1 = Sum(2)
    f1.add(X(), 0, 0)
    f1.add(-X(), 1, 1)
    f1.add(0.5 * Scalar(), 2, [])
    f2 = Sum(2)
    if swap:
        f2.add(X(), 0, 1)
        f2.add(-X(), 1, 0)
    else:
        f2.add(X(), 0, 0)
        f2.add(-X(), 1, 1)
    f2.add(-0.5 * Scalar(), 2, [])
    return f1, f2


def product_of_poissons():
    f1, f2 = home_away(False)
    F = Product(2, 2)
    F.add(Poisson(4) @ Exp() @ f1, [0, 1, 2], [0, 1], 1)
    F.add(Poisson(4) @ Exp() @ f2, [0, 1, 2], [1, 0], 0)
    return F


# observation-indexed models: 9 observations of 3 teams / players
HOME = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
AWAY = np.array([1, 2, 0, 2, 0, 1, 1, 0, 2])
VENUE = rng.normal(0, 1, 9)
W = rng.uniform(.5, 1.5, 9)
P = rng.normal(0, .2, 9)
M = rng.normal(0, .2, (3, 2))
ROW = np.array([0, 1, 2, 2, 1, 0, 0, 1, 2])
COL = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0])


def take_attack_defence():
    # y[o] = a[home[o]] - b[away[o]] + h * venue[o]
    f = Sum(1)
    f.add(Take(HOME), 0, 0)
    f.add(-Take(AWAY), 1, 0)
    f.add(Vector(VENUE), 2, 0)
    return f


def take_same_param():
    # y[o] = a[home[o]] - a[away[o]]: the same parameter gathered twice
    f = Sum(1)
    f.add(Take(HOME), 0, 0)
    f.add(-Take(AWAY), 0, 0)
    return f


def take_and_grid():
    # an indexed effect plus one parameter per observation
    f = Sum(1)
    f.add(Take(HOME), 0, 0)
    f.add(X(), 1, 0)
    return f


def take_product():
    f = Product(1)
    f.add(Take(HOME), 0, 0)
    f.add(Exp() @ Take(AWAY, W), 1, 0)
    return f


def take_two_axes():
    # two indexed effects on different feature axes: y[o, k]
    f = Sum(2)
    f.add(Take(HOME), 0, 0)
    f.add(-Take(COL[:4]), 1, 1)
    return f


def indexed():
    return {
        "Take": (Take(HOME), [U]),
        "Take (weight)": (Take(HOME, W), [U]),
        "Take (negative index)": (Take(HOME - 3), [U]),
        "Take (2 axes)": (Take((ROW, COL)), [M]),
        "Take (index on a grid)": (Take(HOME.reshape(3, 3)), [U]),
        "Affine @ Take": (2. * Take(HOME) + 1., [U]),
        "Sum of Take": (take_attack_defence(), [A, B, H]),
        "Exp @ Sum of Take": (Exp() @ take_attack_defence(), [A, B, H]),
        "Logistic @ Sum of Take": (
            Logistic() @ take_attack_defence(), [A, B, H]),
        "Poisson @ Exp @ Sum of Take": (
            Poisson(6) @ Exp() @ take_attack_defence(), [A, B, H]),
        "NegativeBinomial @ Exp @ Sum of Take": (
            NegativeBinomial(5, 2.) @ Exp() @ take_attack_defence(),
            [A, B, H]),
        "Take, same param": (take_same_param(), [A]),
        "Exp @ Take, same param": (Exp() @ take_same_param(), [A]),
        "Logistic @ Take, same param": (
            Logistic() @ take_same_param(), [A]),
        "Poisson @ Exp @ Take, same param": (
            Poisson(4) @ Exp() @ take_same_param(), [A]),
        "Product of Take": (take_product(), [A, B]),
        "Exp @ Product of Take": (Exp() @ take_product(), [A, B]),
        "Take on two feature axes": (take_two_axes(), [A, V[:2]]),
        "Exp @ Take on two feature axes": (
            Exp() @ take_two_axes(), [A, V[:2]]),
        "Take and X": (take_and_grid(), [A, P]),
        "Poisson @ Exp @ (Take and X)": (
            Poisson(4) @ Exp() @ take_and_grid(), [A, P]),
        "Collapse of Take": (
            Collapse(Exp() @ take_attack_defence()), [A, B, H]),
        "Collapse of Take (dist axes)": (
            Collapse(Poisson(4) @ Exp() @ take_attack_defence(),
                     False, True), [A, B, H]),
        "Collapse of Take (feature axes)": (
            Collapse(Poisson(4) @ Exp() @ take_attack_defence(),
                     True, False), [A, B, H]),
        "Linear @ Exp @ Sum of Take": (
            Linear([1.]) @ Exp() @ take_attack_defence(), [A, B, H]),
        "Take @ Exp (gather from a grid)": (
            Exp() @ Take(HOME) @ Exp() @ X(), [A]),
        "MarkovMatrix of Take": (
            MarkovMatrix(steps=6, size=3) @
            [Exp() @ take_attack_defence(), Exp() @ take_same_param()],
            [A, B, H]),
    }


ELEMENTARY = {
    "X": (X(), [U]),
    "Scalar": (Scalar(), [.7]),
    "Vector": (Vector(V), [.7]),
    "Vector (element-wise)": (Vector(V), [U]),
    "Vector (broadcast v)": (Vector(V), [np.outer(U, V)]),
    "Constant": (Constant(V), [U]),
    "Linear": (Linear([1., 2.]), [U, V]),
    "Linear @ Exp": (Linear([2.]) @ Exp(), [U]),
    "Exp @ Linear": (Exp() @ Linear([1., -1.]), [U, V]),
    "Quadratic": (Quadratic(.3, 2.), [U, V]),
    "Exp": (Exp(), [U]),
    "Log": (Log(), [U]),
    "Logistic": (Sigmoid("logistic"), [U]),
    "ArcTan": (Sigmoid("arctan"), [U]),
    "Poisson": (Poisson(size=6), [U]),
    "LogNegativeBinomial": (LogNegativeBinomial(size=6, r=2.), [U]),
    "NegativeBinomial": (NegativeBinomial(size=6, r=2.), [U]),
    "Affine": (2. * Exp() + 1., [U]),
    "MarkovMatrix": (MarkovMatrix(size=5), [U, V]),
    "MarkovVector": (MarkovVector(size=5), [U, V]),
    # non-uniform skew: the cross derivative H12 used to be wrong
    "MarkovMatrix (skew)": (
        MarkovMatrix(size=4, steps=6,
                     skew=np.random.RandomState(1).uniform(.5, 1.5, (4, 4))),
        [U, V]),
    "MarkovVector (skew)": (
        MarkovVector(size=4, steps=6,
                     skew=np.random.RandomState(2).uniform(.5, 1.5, 7)),
        [U, V]),
}


def composite():
    f1, f2 = home_away(True)
    g1, g2 = home_away(False)
    p1, p2 = home_away(True)

    prod = Product(2)
    prod.add(X(), 0, 0)
    prod.add(X(), 1, 1)

    prod_exp = Product(2)
    prod_exp.add(Exp(), 0, 0)
    prod_exp.add(Exp() @ X(), 1, 1)

    return {
        "Sum (attack/defence)": (attack_defence(), [A, B, H]),
        "Exp @ Sum": (Exp() @ attack_defence(), [A, B, H]),
        "Linear @ Sum": (Linear([1.]) @ attack_defence(), [A, B, H]),
        "Collapse": (Collapse(Exp() @ attack_defence()), [A, B, H]),
        "Product": (prod, [U, V]),
        "Product of Exp": (prod_exp, [U, V]),
        "Poisson @ Exp @ Sum": (
            Poisson(6) @ Exp() @ attack_defence(), [A, B, H]),
        "same param, two axes": (same_param_two_axes(), [A]),
        "Exp @ same param": (Exp() @ same_param_two_axes(), [A]),
        "Logistic @ same param": (
            Logistic() @ same_param_two_axes(), [A]),
        "Poisson @ Exp @ same param": (
            Poisson(4) @ Exp() @ same_param_two_axes(), [A]),
        "MarkovMatrix model": (
            MarkovMatrix(steps=18, size=4) @ [Exp() @ f1, Exp() @ f2],
            [A, B, H]),
        "MarkovVector model": (
            MarkovVector(steps=18, size=4) @ [Exp() @ g1, Exp() @ g2],
            [A, B, H]),
        "CollapseMatrix @ Product": (
            CollapseMatrix() @ product_of_poissons(), [A, B, H]),
        # custom conditions: over / under 2.5 goals
        "CollapseMatrix (over/under) @ Product": (
            CollapseMatrix([(1, 1, -2.5, 1), (1, 1, -2.5, -1)]) @
            product_of_poissons(), [A, B, H]),
        "CollapseMatrix @ MarkovMatrix": (
            CollapseMatrix() @ MarkovMatrix(steps=18, size=4) @
            [Exp() @ p2, Exp() @ p1], [A, B, H]),
    }


class Test(unittest.TestCase):

    def check(self, cases):
        for name, (foo, params) in cases.items():
            for use_eval in (True, False):
                with self.subTest(name, use_eval=use_eval):
                    self.assertEqual(
                        derivative_errors(foo, params, use_eval=use_eval,
                                          rtol=1e-4, atol=1e-6), [])

    def test_elementary(self):
        self.check(ELEMENTARY)

    def test_composite(self):
        self.check(composite())

    def test_indexed(self):
        self.check(indexed())


if __name__ == "__main__":
    unittest.main()
