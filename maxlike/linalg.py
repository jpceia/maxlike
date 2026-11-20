"""
Direct solver for the sparse, symmetric Newton (KKT) system of a problem
with observation-indexed effects.

In such a problem the Hessian block of a set of effects with respect to
itself is often diagonal: an observation involves one attacking team, one
student, one origin, ... so two entries of that parameter never meet. Those
entries are eliminated exactly (block elimination on the diagonal block D):

    | D   B' | |x_e|   |b_e|        S = C - B D^-1 B'   (Schur complement)
    | B   C  | |x_k| = |b_k|        S x_k = b_k - B D^-1 b_e
                                    x_e = D^-1 (b_e - B' x_k)

and only S, over the remaining entries, is factorised: as a dense matrix
(LAPACK) when it is small or not sparse enough, else with a sparse LU
(SuperLU). For students x questions data, say, S has the size of the number
of questions, whatever the number of students.

Limits: memory and time are those of the factorisation of S. When effects
meet at random (every team plays random opponents) S, or its sparse factor,
fills in, and the cost is that of a dense matrix of its size.
"""
import re
import warnings
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import reverse_cuthill_mckee

# eliminate a diagonal block only if it has at least this share of the rows
ELIMINATION_MIN_SHARE = 0.2
# the matrix to factorise is treated as dense up to this size, and above it
# when a sparse factor would fill at least the share DENSE_MIN_FILL of the
# matrix (see fill_estimate) and it has at most DENSE_MAX_SIZE rows
# (8 * size^2 bytes for the factor)
DENSE_SIZE = 3000
DENSE_MIN_FILL = 0.15
# ... or when its density is at least this (no estimate needed)
DENSE_MIN_DENSITY = 0.05
DENSE_MAX_SIZE = 20000
# a matrix at least this dense is factorised as it is, as a dense matrix
DENSE_DIRECT_DENSITY = 0.25



def _version(text):
    return tuple(int(x) for x in re.findall(r"\d+", text)[:2])


# SciPy 1.7's SuperLU crashed intermittently (segmentation fault inside
# splu) in the minimum-dependencies CI job, on both sparse and nearly
# dense matrices; it never did with current SciPy. With older SciPy the
# sparse LU is not used up to DENSE_MAX_SIZE rows (dense LAPACK instead).
SPARSE_LU_SAFE = _version(scipy.__version__) >= (1, 8)


class SingularMatrix(np.linalg.LinAlgError):
    """The matrix is exactly singular."""


def fill_estimate(S):
    """
    Rough share of a symmetric sparse matrix that its factorisation fills:
    the envelope of the matrix (from the first non-zero of each row to the
    diagonal) after a reverse Cuthill-McKee ordering, which bounds the fill
    of a banded factorisation. The few rows that are much denser than the
    others (a parameter shared by all the observations, a constraint) are
    left out, as an ordering puts them last.

    It is small for matrices with local structure (leagues, regions,
    neighbours) and close to 1 when the entries are scattered at random,
    whatever their number: the factor of such a matrix is nearly full.
    """
    S = sp.csr_matrix(S)
    m = S.shape[0]
    if m == 0:
        return 0.
    count = np.diff(S.indptr)
    light = np.flatnonzero(count <= max(10 * np.median(count), 10))
    S = S[light][:, light]
    S = sp.csr_matrix(S + S.T)
    perm = reverse_cuthill_mckee(S, symmetric_mode=True)
    S = sp.csr_matrix(S[perm][:, perm])
    S.sort_indices()
    rows = np.flatnonzero(np.diff(S.indptr))
    first = S.indices[S.indptr[rows]]
    envelope = np.maximum(rows - first, 0).sum() + (m - light.size) * m
    return envelope / (m * m / 2.)


class SymmetricSolver:
    """
    Factorisation of a sparse symmetric matrix A, to solve A x = b and to
    get the diagonal of its inverse.

    Parameters
    ----------
    A : scipy.sparse matrix (symmetric)
    blocks : list of (start, stop)
        Ranges of rows that are candidates for elimination (the entries of
        each parameter); a range is eliminated if A is diagonal there.
    """

    def __init__(self, A, blocks=()):
        A = sp.csr_matrix(A)
        self.size = A.shape[0]
        if self.size <= DENSE_MAX_SIZE and \
                A.nnz >= DENSE_DIRECT_DENSITY * self.size ** 2:
            # nothing to gain from the structure
            self.elim = np.zeros(0, dtype=np.intp)
        else:
            self.elim = self._diagonal_rows(A, blocks)
        if self.elim.size < ELIMINATION_MIN_SHARE * self.size:
            self.elim = self.elim[:0]
        if self.elim.size:
            keep = np.ones(self.size, dtype=bool)
            keep[self.elim] = False
            self.keep = np.flatnonzero(keep)
            self.d = A.diagonal()[self.elim]
            rows = A[self.keep]
            self.B = sp.csr_matrix(rows[:, self.elim])
            A = rows[:, self.keep] - \
                sp.csr_matrix(self.B.multiply(1 / self.d)) @ self.B.T
        self._factor(sp.csr_matrix(A))

    @staticmethod
    def _diagonal_rows(A, blocks):
        """
        Rows to eliminate: the largest candidate range where A is diagonal,
        plus the other diagonal ranges that are not coupled with it.
        """
        diag = A.diagonal()
        tiny = 1e-12 * np.abs(diag).max() if diag.size else 0.
        chosen = np.zeros(0, dtype=np.intp)
        for start, stop in sorted(blocks, key=lambda b: b[0] - b[1]):
            if stop - start < 2:
                continue
            rows = A[start:stop]
            if rows[:, start:stop].count_nonzero() != \
                    np.count_nonzero(diag[start:stop]):
                continue
            idx = start + np.flatnonzero(np.abs(diag[start:stop]) > tiny)
            if chosen.size and A[idx][:, chosen].count_nonzero():
                continue
            chosen = np.concatenate([chosen, idx])
        return np.sort(chosen)

    def _factor(self, S):
        m = S.shape[0]
        # a matrix that is already nearly dense is always factorised dense:
        # LAPACK is faster there, and SuperLU on such matrices crashed
        # intermittently (segmentation fault) with old SciPy (1.7)
        nearly_dense = m <= DENSE_MAX_SIZE and \
            S.nnz >= DENSE_DIRECT_DENSITY * m * m
        old_scipy = not SPARSE_LU_SAFE and m <= DENSE_MAX_SIZE
        self.dense = m <= DENSE_SIZE or nearly_dense or old_scipy or (
            m <= DENSE_MAX_SIZE and DENSE_MIN_FILL <= 1 and
            (S.nnz >= DENSE_MIN_DENSITY * m * m or
             fill_estimate(S) >= DENSE_MIN_FILL))
        if self.dense:
            with warnings.catch_warnings():
                # an exactly singular matrix gives a non-finite solution
                warnings.simplefilter("ignore", LinAlgWarning)
                # Fortran order: factorised in place, without a copy
                self.lu = lu_factor(S.toarray(order="F"), overwrite_a=True,
                                    check_finite=False)
            if not np.all(np.diag(self.lu[0]) != 0):
                raise SingularMatrix
        else:
            try:
                # symmetric matrix: fill-reducing ordering of A' + A
                self.lu = splu(S.tocsc(), permc_spec="MMD_AT_PLUS_A")
            except RuntimeError:
                raise SingularMatrix

    def _solve_reduced(self, b):
        if self.dense:
            return lu_solve(self.lu, b, check_finite=False)
        return self.lu.solve(b)

    def solve(self, b):
        """Solution of A x = b, for a vector or a matrix of columns b"""
        b = np.asarray(b, dtype=float)
        if not self.elim.size:
            return self._solve_reduced(b)
        d = self.d if b.ndim == 1 else self.d[:, None]
        b_e = b[self.elim]
        x = np.empty(b.shape)
        x_k = self._solve_reduced(b[self.keep] - self.B @ (b_e / d))
        x[self.keep] = x_k
        x[self.elim] = (b_e - self.B.T @ x_k) / d
        return x

    def inverse_diagonal(self, batch_bytes=1 << 25):
        """
        Diagonal of the inverse of A. The solver cannot be used afterwards.

        For the entries kept, diag(S^-1); for an eliminated entry e,
        1 / d_e + (b_e' S^-1 b_e) / d_e^2 with b_e its column of B. A dense
        S is inverted; with a sparse factorisation there is one solve per
        row of A, a batch of columns at a time.
        """
        out = np.empty(self.size)
        keep = self.keep if self.elim.size else np.arange(self.size)
        m = keep.size
        batch = max(1, min(max(m, 1), batch_bytes // (8 * max(m, 1))))

        if self.dense:
            inv = lu_solve(self.lu, np.eye(m, order="F"), overwrite_b=True,
                           check_finite=False)
            self.lu = None      # frees the memory of the factorisation
            out[keep] = np.diag(inv)

            def quad(cols):
                # cols: columns of B (sparse, m x batch)
                return np.asarray(cols.multiply((cols.T @ inv).T).sum(0)).ravel()
        else:
            for start in range(0, m, batch):
                idx = np.arange(start, min(start + batch, m))
                rhs = np.zeros((m, idx.size))
                rhs[idx, np.arange(idx.size)] = 1
                out[keep[idx]] = self.lu.solve(rhs)[idx, np.arange(idx.size)]

            def quad(cols):
                return np.asarray(cols.multiply(
                    self.lu.solve(cols.toarray())).sum(0)).ravel()

        if self.elim.size:
            B = self.B.tocsc()
            q = np.empty(self.elim.size)
            for start in range(0, q.size, batch):
                stop = min(start + batch, q.size)
                q[start:stop] = quad(B[:, start:stop])
            out[self.elim] = 1 / self.d + q / (self.d * self.d)
        return out
