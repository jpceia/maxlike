[![PyPI version](https://badge.fury.io/py/maxlike.svg)](https://badge.fury.io/py/maxlike)
[![tests](https://github.com/jpceia/maxlike/actions/workflows/tests.yml/badge.svg)](https://github.com/jpceia/maxlike/actions/workflows/tests.yml)

# MaxLike

**MaxLike fits statistical models with Newton's method using exact Hessians that exploit sparsity.**

In models with indexed effects (teams, players, items, states, …) almost all
second derivatives are zero: the rate of *team A vs team B* depends only on
`attack[A]` and `defence[B]`, not on the other teams. MaxLike propagates first
and second derivatives through the model *keeping that structure*: a derivative
that is diagonal in the parameter/feature axes is stored only by its diagonal.
The full dense Jacobian and Hessian of the model are never materialised.

This gives you:

- **Exact Newton steps** with quadratic convergence, instead of first-order
  methods or quasi-Newton approximations;
- **Standard errors, covariance and Fisher information for free**, since the
  Hessian is already there at the optimum;
- **Low memory and time** where general-purpose autodiff would compute dense
  Hessians column by column.

Example: the attack/defence Poisson model below, with `n` teams. The Jacobian
stored by MaxLike is compared with its dense equivalent, and the fit time with
8 Newton steps using `jax.hessian` under `jit` (one CPU, excluding JIT
compilation):

| teams | parameters | Jacobian stored | dense Jacobian | MaxLike (full fit) | JAX Newton |
|------:|-----------:|----------------:|---------------:|-------------------:|-----------:|
|    20 |         41 |              42 |         32 800 |             0.01 s |     0.24 s |
|   100 |        201 |             202 |      4 020 000 |             0.03 s |     0.39 s |
|   200 |        401 |             402 |     32 080 000 |             0.09 s |     1.61 s |
|   400 |        801 |             802 |    256 320 000 |             0.33 s |     8.98 s |

Both reach the same estimates.

## Installation

```bash
pip install maxlike
```

Requires Python 3.10+, NumPy, SciPy and pandas.

## Example: team strengths from match scores

```python
import numpy as np
import maxlike
from maxlike.func import X, Vector, Sum, Linear

# Goals scored by team i (at home or away) against team j:
#   log E[goals] = attack[i] - defence[j] + home * (h - 1/2)
n_teams = 20
rng = np.random.default_rng(0)
attack, defence = rng.normal(0, .3, (2, n_teams))
h = np.arange(2) - .5                       # 0: away, 1: home
rate = np.exp(attack[:, None, None] - defence[None, :, None] + .25 * h)

N = np.ones((n_teams, n_teams, 2))          # number of games per cell
np.einsum('iik->ik', N)[:] = 0              # a team never plays itself
goals = rng.poisson(rate) * N               # observed totals per cell

mle = maxlike.Poisson()
mle.model = Sum(3)                          # 3 feature axes: (i, j, h)
mle.model.add(X(), 0, 0)                    # param 0 (attack)  -> axis 0
mle.model.add(-X(), 1, 1)                   # param 1 (defence) -> axis 1
mle.model.add(Vector(h), 2, 2)              # param 2 (home)    -> axis 2
mle.add_constraint([0, 1], Linear([1, 1]))  # identifiability: sum = 0

mle.add_param(np.zeros(n_teams))
mle.add_param(np.zeros(n_teams))
mle.add_param(0.)

mle.fit(N=N, X=goals)

a, d, home = mle.get_params()
s_a, s_d, s_home = mle.std_error()
print("home advantage: %.3f +/- %.3f" % (home, s_home))
print("AIC: %.4f" % mle.akaike_information_criterion())
```

```
home advantage: 0.177 +/- 0.073
AIC: 2.5834
```

## How it works

A problem has three independent parts.

1. **Model `y = f(θ)`** (`maxlike.func`). It is built by composing functions:
   `Sum`, `Product`, `Compose` (`f @ g`), `X`, `Vector`, `Exp`, `Log`, the
   sigmoids, `MarkovMatrix`, copulas and so on. Each function returns its
   value, gradient and Hessian with respect to every parameter block.
   `Sum.add(f, param_map, feat_map)` says which parameters `f` uses and which
   feature axes of the output it maps to.
2. **Score `L(y)`**, the objective to maximise, given its first and second
   derivatives with respect to `y`:
   - `Poisson`, `ZeroInflatedPoisson` and `NegativeBinomial` for counts;
   - `Logistic` for binary data;
   - `Finite` and `NormalizedFinite` for arbitrary discrete distributions,
     which also minimise a Kullback–Leibler divergence when frequencies are
     replaced by probabilities.

   Maximum likelihood is the main use, but any score that is a sum over cells
   of `y` fits the same scheme.
3. **Solver** (`MaxLike.fit`). It runs a damped Newton method (or Broyden)
   with equality constraints through Lagrange multipliers, optional
   regularisation, and parameters that can be partially fixed.

### Structured tensors

The derivatives are `maxlike.Tensor` objects. Their axes are split into four
groups:

| axes    | meaning                                                |
|---------|--------------------------------------------------------|
| `x_dim` | axes of parameter *i* (1st derivative)                 |
| `y_dim` | axes of parameter *j* (2nd derivative)                 |
| `n`     | feature axes (e.g. home team, away team, venue)        |
| `dim`   | distribution axes (e.g. number of goals `0..k`)        |

`x_map` and `y_map` record that a parameter axis coincides with a feature axis,
meaning the derivative is diagonal there and only the diagonal is stored. The
chain-rule contractions (`dot_left` and `dot_right`) and the arithmetic keep
these maps. `GenericTensor` holds a lazy sum of terms with incompatible
structure instead of densifying them. `Compose` applies the full second-order
chain rule, `H = Jᵀ H_f J + Σ_k ∇f_k H_{g_k}`, on these structured objects.

## Limitations

- CPU only (NumPy); no GPU support.
- The final Newton system over the free parameters is dense, so the method
  suits problems up to a few thousand parameters.
- The built-in scores assume the objective is a sum over the cells of `y`
  (`Finite` additionally normalises along the distribution axes).

## Development

```bash
pip install -e ".[test]"
pytest
```
