"""
Observation-indexed models (maxlike.func.Take) against the equivalent grid
models on the same data: same estimates and standard errors, with the dense
and with the sparse assembly of the Hessian.
"""
import unittest
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import maxlike
from maxlike import linalg
from maxlike.func import (
    X, Take, Vector, Sum, Linear, Quadratic, Exp, Poisson)
from maxlike.preprocessing import prepare_index, prepare_observations


def football_data(n, n_games, seed=0):
    """One row per (attacking team, defending team, venue) of each game"""
    rng = np.random.RandomState(seed)
    a, b = rng.normal(0, .3, (2, n))
    # every team meets its neighbour, so the graph of games is connected
    home = np.concatenate([np.arange(n), rng.randint(0, n, n_games)])
    away = np.concatenate([(np.arange(n) + 1) % n,
                           rng.randint(0, n, n_games)])
    home, away = home[home != away], away[home != away]
    att = np.concatenate([home, away])
    dfn = np.concatenate([away, home])
    venue = np.repeat([1, 0], home.size)
    goals = rng.poisson(np.exp(a[att] - b[dfn] + .3 * (venue - .5)))
    return att, dfn, venue, goals.astype(float)


def to_grid(shape, index, values):
    grid = np.zeros(shape)
    np.add.at(grid, index, values)
    return grid


def grid_model():
    f = Sum(3)
    f.add(X(), 0, 0)
    f.add(-X(), 1, 1)
    f.add(Vector(np.arange(2) - .5), 2, 2)
    return f


def indexed_model(att, dfn, venue):
    f = Sum(1)
    f.add(Take(att), 0, 0)
    f.add(-Take(dfn), 1, 0)
    f.add(Vector(venue - .5), 2, 0)
    return f


def football(mle, model, n, fixed=None):
    mle.model = model
    mle.add_constraint([0, 1], Linear([1, 1]))
    mle.add_param(np.zeros(n))
    mle.add_param(np.zeros(n), fixed)
    mle.add_param(0.)
    return mle


class Test(unittest.TestCase):

    def setUp(self):
        self._settings = (maxlike.maxlike_base.MaxLike.sparse_hess,
                          linalg.DENSE_SIZE, linalg.DENSE_MIN_FILL,
                          linalg.DENSE_DIRECT_DENSITY)

    def tearDown(self):
        (maxlike.maxlike_base.MaxLike.sparse_hess, linalg.DENSE_SIZE,
         linalg.DENSE_MIN_FILL, linalg.DENSE_DIRECT_DENSITY) = \
            self._settings

    def same_fit(self, a, b, tol=1e-8, matrices=True):
        for p, q in zip(a.get_params(), b.get_params()):
            np.testing.assert_allclose(p.filled(0), q.filled(0), atol=tol)
        for p, q in zip(a.std_error(), b.std_error()):
            np.testing.assert_allclose(p.filled(0), q.filled(0), atol=tol)
        if matrices:
            for m in ("fisher_matrix", "error_matrix"):
                for row_a, row_b in zip(getattr(a, m)(), getattr(b, m)()):
                    for p, q in zip(row_a, row_b):
                        np.testing.assert_allclose(
                            np.ma.filled(p, 0), np.ma.filled(q, 0),
                            atol=1e-6)

    def both_assemblies(self, build, reference, tol=1e-8):
        """Fits build() with the dense and the sparse Hessian"""
        for sparse, kind in ((False, np.ndarray), (True, sp.spmatrix)):
            with self.subTest(sparse=sparse):
                maxlike.maxlike_base.MaxLike.sparse_hess = sparse
                mle = build()
                self.assertIsInstance(mle.flat_hess_, kind)
                self.same_fit(mle, reference, tol)
        maxlike.maxlike_base.MaxLike.sparse_hess = None

    # ----------------------------------------------------------------------
    # football: attack / defence Poisson
    # ----------------------------------------------------------------------
    def test_poisson(self):
        n = 12
        att, dfn, venue, goals = football_data(n, 150)
        ones = np.ones(goals.size)
        fixed = np.arange(n) == 3

        grid = football(maxlike.Poisson(), grid_model(), n, fixed)
        grid.fit(N=to_grid((n, n, 2), (att, dfn, venue), ones),
                 X=to_grid((n, n, 2), (att, dfn, venue), goals))

        def build():
            mle = football(maxlike.Poisson(),
                           indexed_model(att, dfn, venue), n, fixed)
            mle.fit(N=ones, X=goals)
            return mle

        self.both_assemblies(build, grid)

    def test_poisson_aggregated(self):
        # repeated observations added up: same fit with fewer rows
        n = 8
        att, dfn, venue, goals = football_data(n, 300, seed=1)
        df = pd.DataFrame({"att": att, "dfn": dfn, "venue": venue,
                           "goals": goals}).astype({"att": str, "dfn": str})
        index, labels, data = prepare_observations(
            df, {"team": ["att", "dfn"]}, "goals", by="venue")
        self.assertLess(data["N"].size, goals.size)
        self.assertEqual(data["N"].sum(), goals.size)
        self.assertEqual(list(labels["team"]), sorted(map(str, range(n))))

        full = football(maxlike.Poisson(),
                        indexed_model(att, dfn, venue), n)
        full.fit(N=np.ones(goals.size), X=goals)

        agg = maxlike.Poisson()
        agg.model = indexed_model(index["att"], index["dfn"], data["venue"])
        agg.add_constraint([0, 1], Linear([1, 1]))
        agg.add_param(np.zeros(n), name="attack", labels=labels["team"])
        agg.add_param(np.zeros(n), name="defence", labels=labels["team"])
        agg.add_param(0., name="home")
        agg.fit(N=data["N"], X=data["goals"])

        # the parameters of `agg` follow the order of the string labels
        order = np.array([int(t) for t in labels["team"]])
        for k in range(2):
            np.testing.assert_allclose(agg.get_params()[k],
                                       full.get_params()[k][order],
                                       atol=1e-8)
            np.testing.assert_allclose(agg.std_error()[k],
                                       full.std_error()[k][order],
                                       atol=1e-8)
        self.assertEqual(agg.summary().shape[0], 2 * n + 1)

    def test_regularization(self):
        n = 10
        att, dfn, venue, goals = football_data(n, 80, seed=2)
        ones = np.ones(goals.size)

        def problem(model):
            mle = maxlike.Poisson()
            mle.model = model
            mle.add_regularization([0, 1], Quadratic(0, 1.5))
            mle.add_param(np.zeros(n))
            mle.add_param(np.zeros(n))
            mle.add_param(0.)
            return mle

        grid = problem(grid_model())
        grid.fit(N=to_grid((n, n, 2), (att, dfn, venue), ones),
                 X=to_grid((n, n, 2), (att, dfn, venue), goals))

        def build():
            mle = problem(indexed_model(att, dfn, venue))
            mle.fit(N=ones, X=goals)
            return mle

        self.both_assemblies(build, grid)

    def test_negative_binomial(self):
        n = 9
        att, dfn, venue, goals = football_data(n, 120, seed=3)
        ones = np.ones(goals.size)
        grid = football(maxlike.NegativeBinomial(2.5), grid_model(), n)
        # the fit stops on the change of the objective: a tight tolerance
        # brings the parameters within 1e-8 of each other
        grid.fit(N=to_grid((n, n, 2), (att, dfn, venue), ones),
                 X=to_grid((n, n, 2), (att, dfn, venue), goals), tol=1e-13)

        def build():
            mle = football(maxlike.NegativeBinomial(2.5),
                           indexed_model(att, dfn, venue), n)
            mle.fit(N=ones, X=goals, tol=1e-13)
            return mle

        self.both_assemblies(build, grid)

    # ----------------------------------------------------------------------
    # distribution axis: Poisson pmf with the Finite likelihood
    # ----------------------------------------------------------------------
    def test_finite_poisson_pmf(self):
        n, size = 7, 8
        att, dfn, venue, goals = football_data(n, 90, seed=4)
        goals = np.minimum(goals, size - 1).astype(int)
        counts = np.zeros((goals.size, size))
        counts[np.arange(goals.size), goals] = 1

        grid = football(maxlike.Finite(dim=1),
                        Poisson(size) @ Exp() @ grid_model(), n)
        grid.fit(N=to_grid((n, n, 2, size), (att, dfn, venue, goals), 1.))

        def build():
            mle = football(
                maxlike.Finite(dim=1),
                Poisson(size) @ Exp() @ indexed_model(att, dfn, venue), n)
            mle.fit(N=counts)
            return mle

        self.both_assemblies(build, grid)

    # ----------------------------------------------------------------------
    # tennis: Bradley-Terry, the same parameter gathered twice
    # ----------------------------------------------------------------------
    def test_bradley_terry(self):
        n = 10
        rng = np.random.RandomState(5)
        skill = rng.normal(0, .7, n)
        p1 = np.concatenate([np.arange(n), rng.randint(0, n, 200)])
        p2 = np.concatenate([(np.arange(n) + 1) % n,
                             rng.randint(0, n, 200)])
        p1, p2 = p1[p1 != p2], p2[p1 != p2]
        wins = (rng.uniform(size=p1.size) <
                1 / (1 + np.exp(skill[p2] - skill[p1]))).astype(float)
        ones = np.ones(p1.size)

        grid = maxlike.Logistic()
        grid.model = Sum(2)
        grid.model.add(X(), 0, 0)
        grid.model.add(-X(), 0, 1)
        grid.add_constraint(0, Linear([1]))
        grid.add_param(np.zeros(n))
        grid.fit(N=to_grid((n, n), (p1, p2), ones),
                 X=to_grid((n, n), (p1, p2), wins))

        def build():
            mle = maxlike.Logistic()
            mle.model = Sum(1)
            mle.model.add(Take(p1), 0, 0)
            mle.model.add(-Take(p2), 0, 0)
            mle.add_constraint(0, Linear([1]))
            mle.add_param(np.zeros(n))
            mle.fit(N=ones, X=wins)
            return mle

        self.both_assemblies(build, grid)

    # ----------------------------------------------------------------------
    # sparse Hessian and solver
    # ----------------------------------------------------------------------
    def test_sparse_by_default_when_large(self):
        n = 300
        att, dfn, venue, goals = football_data(n, 3000, seed=6)
        ones = np.ones(goals.size)

        def build():
            mle = football(maxlike.Poisson(),
                           indexed_model(att, dfn, venue), n,
                           np.arange(n) % 50 == 7)
            mle.fit(N=ones, X=goals)
            return mle

        auto = build()
        self.assertTrue(sp.issparse(auto.flat_hess_))
        maxlike.maxlike_base.MaxLike.sparse_hess = False
        dense = build()
        self.assertIsInstance(dense.flat_hess_, np.ndarray)
        np.testing.assert_allclose(auto.flat_hess_.toarray(),
                                   dense.flat_hess_, atol=1e-9)
        self.same_fit(auto, dense, matrices=False)

        # the same with the sparse LU of the Schur complement, and with the
        # sparse LU of the whole matrix
        maxlike.maxlike_base.MaxLike.sparse_hess = True
        linalg.DENSE_SIZE = 0
        linalg.DENSE_MIN_FILL = 2.
        self.same_fit(build(), dense, matrices=False)
        elimination = linalg.ELIMINATION_MIN_SHARE
        try:
            linalg.ELIMINATION_MIN_SHARE = 2.
            self.same_fit(build(), dense, matrices=False)
        finally:
            linalg.ELIMINATION_MIN_SHARE = elimination

    def test_grid_model_with_sparse_hessian(self):
        # sparse_hess=True also applies to a grid model (dense blocks)
        n = 6
        att, dfn, venue, goals = football_data(n, 60, seed=7)
        data = dict(
            N=to_grid((n, n, 2), (att, dfn, venue), np.ones(goals.size)),
            X=to_grid((n, n, 2), (att, dfn, venue), goals))
        dense = football(maxlike.Poisson(), grid_model(), n)
        dense.fit(**data)
        self.assertIsInstance(dense.flat_hess_, np.ndarray)
        sparse = football(maxlike.Poisson(), grid_model(), n)
        sparse.sparse_hess = True
        sparse.fit(**data)
        self.assertTrue(sp.issparse(sparse.flat_hess_))
        self.same_fit(sparse, dense)

    def test_not_identifiable(self):
        # no constraint: attack and defence can be shifted together
        n = 6
        att, dfn, venue, goals = football_data(n, 60, seed=8)
        for sparse in (False, True):
            mle = maxlike.Poisson()
            mle.sparse_hess = sparse
            mle.model = indexed_model(att, dfn, venue)
            mle.add_param(np.zeros(n))
            mle.add_param(np.zeros(n))
            mle.add_param(0.)
            mle.N = maxlike.Tensor(np.ones(goals.size))
            mle.X = maxlike.Tensor(goals)
            self.assertEqual(
                mle.identifiability(),
                ["all of 'param 0' (+1) + all of 'param 1' (+1)"])
            # fit itself explains it, on the dense and the sparse path
            with self.assertRaisesRegex(maxlike.IdentifiabilityError,
                                        "all of 'param 0'"):
                mle.fit(N=np.ones(goals.size), X=goals)

    def test_broyden_with_sparse_hessian(self):
        n = 30
        att, dfn, venue, goals = football_data(n, 300, seed=12)
        fits = {}
        for method in ("newton", "broyden"):
            mle = football(maxlike.Poisson(), indexed_model(att, dfn, venue),
                           n)
            mle.sparse_hess = True
            self.assertTrue(mle.fit(N=np.ones(goals.size), X=goals,
                                    method=method, max_steps=200,
                                    tol=1e-14))
            fits[method] = np.concatenate(
                [np.ravel(np.ma.getdata(p)) for p in mle.get_params()])
        np.testing.assert_allclose(fits["newton"], fits["broyden"],
                                   atol=1e-6)

    def test_predict_guard_before_densifying(self):
        # 2001 free parameters x 30 000 observations: refuse before
        # allocating the dense Jacobian
        n = 1000
        att, dfn, venue, goals = football_data(n, 15000, seed=13)
        mle = football(maxlike.Poisson(), indexed_model(att, dfn, venue), n)
        with self.assertRaisesRegex(ValueError, "Jacobian is too large"):
            mle.predict(se=True)

    def test_solver(self):
        rng = np.random.RandomState(9)
        n_e, n_k = 40, 12
        B = sp.random(n_k, n_e, .2, random_state=rng).toarray()
        C = rng.normal(size=(n_k, n_k))
        A = np.block([[-np.diag(rng.uniform(1, 2, n_e)), B.T],
                      [B, C + C.T]])
        b = rng.normal(size=n_e + n_k)
        rhs = rng.normal(size=(n_e + n_k, 3))
        blocks = [(0, n_e), (n_e, n_e + n_k)]
        for dense_size, share in ((10 ** 6, .2), (0, .2), (0, 2.)):
            with self.subTest(dense_size=dense_size, share=share):
                linalg.DENSE_SIZE = dense_size
                linalg.DENSE_MIN_FILL = 2.
                linalg.DENSE_DIRECT_DENSITY = 2.
                elimination = linalg.ELIMINATION_MIN_SHARE
                try:
                    linalg.ELIMINATION_MIN_SHARE = share
                    solver = linalg.SymmetricSolver(sp.csc_matrix(A), blocks)
                finally:
                    linalg.ELIMINATION_MIN_SHARE = elimination
                self.assertEqual(solver.elim.size, n_e if share < 1 else 0)
                # with SciPy older than 1.8 the sparse LU is never used
                self.assertEqual(solver.dense, dense_size > 0 or
                                 not linalg.SPARSE_LU_SAFE)
                np.testing.assert_allclose(solver.solve(b),
                                           np.linalg.solve(A, b), atol=1e-10)
                np.testing.assert_allclose(solver.solve(rhs),
                                           np.linalg.solve(A, rhs),
                                           atol=1e-10)
                np.testing.assert_allclose(solver.inverse_diagonal(),
                                           np.diag(np.linalg.inv(A)),
                                           atol=1e-10)
        with self.assertRaises(linalg.SingularMatrix):
            linalg.SymmetricSolver(sp.csc_matrix((5, 5)))

    def test_no_sparse_lu_with_old_scipy(self):
        self.assertEqual(linalg._version("1.7.3"), (1, 7))
        self.assertEqual(linalg._version("1.18.1"), (1, 18))
        self.assertEqual(linalg._version("1.10.0rc1"), (1, 10))
        A = sp.random(30, 30, .1, random_state=np.random.RandomState(3))
        A = sp.csc_matrix(A + A.T + 10 * sp.eye(30))
        safe = linalg.SPARSE_LU_SAFE
        linalg.DENSE_SIZE = 0
        linalg.DENSE_MIN_FILL = 2.
        try:
            linalg.SPARSE_LU_SAFE = False
            solver = linalg.SymmetricSolver(A)
            self.assertTrue(solver.dense)
            b = np.arange(30.)
            np.testing.assert_allclose(solver.solve(b),
                                       np.linalg.solve(A.toarray(), b))
        finally:
            linalg.SPARSE_LU_SAFE = safe

    def test_fill_estimate(self):
        # neighbours only (in a shuffled order), plus one dense row and
        # column: a sparse factor stays sparse
        m = 400
        rng = np.random.RandomState(12)
        band = sp.diags([1.] * 7, range(-3, 4), shape=(m, m)).tolil()
        band[0, :] = 1
        band[:, 0] = 1
        perm = rng.permutation(m)
        self.assertLess(linalg.fill_estimate(band.tocsr()[perm][:, perm]),
                        .05)
        # the same number of entries scattered at random: it fills
        scattered = sp.random(m, m, 7. / m, random_state=rng)
        self.assertGreater(
            linalg.fill_estimate(scattered + scattered.T + sp.eye(m)), .5)

    # ----------------------------------------------------------------------
    # around the fit
    # ----------------------------------------------------------------------
    def test_inference(self):
        n = 8
        att, dfn, venue, goals = football_data(n, 100, seed=10)
        mle = football(maxlike.Poisson(), indexed_model(att, dfn, venue), n)
        mle.fit(N=np.ones(goals.size), X=goals)
        value, se = mle.predict(se=True)
        self.assertEqual(value.shape, goals.shape)
        self.assertEqual(se.shape, goals.shape)
        self.assertTrue(np.all(se > 0))
        sim = mle.simulate(0)
        self.assertEqual(sim["X"].shape, goals.shape)
        self.assertEqual(mle.summary().shape[0], 2 * n + 1)
        self.assertEqual(mle.identifiability(), [])

    def test_graph(self):
        n = 8
        att, dfn, venue, goals = football_data(n, 30, seed=11)
        mle = football(maxlike.Poisson(), indexed_model(att, dfn, venue), n)
        labels = [label for _, label, _ in mle.graph().nodes]
        self.assertIn("Take(n_obs=%d, size=%d)" % (att.size, att.max() + 1),
                      labels)
        mle.fit(N=np.ones(goals.size), X=goals)
        labels = [label for _, label, _ in mle.graph().nodes]
        self.assertEqual(
            labels.count("Take(n_obs=%d, size=%d)" % (att.size, n)), 2)
        self.assertIn("Take(n_obs=3, size=2, weighted)", [
            label for _, label, _ in
            Take([0, 1, 1], [1., 2., 3.]).graph().nodes])

    def test_prepare_index(self):
        df = pd.DataFrame({"home": ["b", "a", "c"], "away": ["c", "b", "a"],
                           "ref": [10, 30, 10]})
        index, labels = prepare_index(
            df, {"team": ["home", "away"], "ref": "ref"})
        self.assertEqual(list(labels["team"]), ["a", "b", "c"])
        self.assertEqual(list(labels["ref"]), [10, 30])
        np.testing.assert_array_equal(index["home"], [1, 0, 2])
        np.testing.assert_array_equal(index["away"], [2, 1, 0])
        np.testing.assert_array_equal(index["ref"], [0, 1, 0])
        df.loc[0, "home"] = None
        with self.assertRaises(ValueError):
            prepare_index(df, {"team": ["home", "away"]})


if __name__ == "__main__":
    unittest.main()
