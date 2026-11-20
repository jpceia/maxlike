# Changelog

## Unreleased

### Added
- Parameter names and labels (`add_param(..., name=, labels=)`, taken from
  the index of a pandas Series), `summary()` with standard errors,
  p-values and confidence intervals, and `get_params(labeled=True)`.
- `predict(model, se=True)`: predictions of any function of the
  parameters with delta-method standard errors.
- `simulate()`: data simulated from the fitted model, ready for
  `fit(**...)` (parametric bootstrap).
- `IdentifiabilityError` and `identifiability()`: name the parameters along
  which the objective is flat when a constraint is missing.
- A warning for separated data in logistic models.
- `LogLikelihood`: fit models that return the log-likelihood of each
  observation.
- `FirstOrderStatistic`: the smallest of several latent values wins and
  only the winner and the winning value are observed (auctions, races,
  competing risks), optionally with a latent common factor per
  observation. Documented in `docs/first_order_statistic.md`.
- `func.Poisson(size, normalised=True, tail=...)`: probabilities instead of
  the unnormalised lambda^x / x!, with a censored ("size - 1 or more"),
  truncated or dropped tail.
- Runnable examples in `docs/examples`: football goals, football score
  distribution and betting markets, calibration to market 1X2
  probabilities, tennis, and a race; `tests/examples_test.py` runs them.
- Observation-indexed models: `func.Take(index)` gathers one parameter
  entry per observation, so data listed one row per observation need no
  grid of all the feature combinations. It works with every likelihood,
  with compositions, constraints and regularization, and the same
  parameter can be gathered twice (Bradley–Terry).
- `preprocessing.prepare_index` and `prepare_observations` build the index
  arrays and labels from a DataFrame with one row per observation.
- Sparse Newton system for observation-indexed models with 500 or more
  free parameters (`mle.sparse_hess` forces the choice): the Hessian is a
  `scipy.sparse` matrix, solved by `maxlike.linalg.SymmetricSolver`.
  `std_error()` computes only the diagonal of the inverse. Attack/defence
  model with 5 000 teams and 100 000 rows on one thread: 10 s and 0.3 GB,
  against 130 s and 4.6 GB on the grid.

### Fixed
- The convergence test no longer divides by the objective; refitting from
  the optimum no longer raises a ConvergenceError.
- Newton steps that would decrease the objective (Hessian not negative
  definite) are damped Levenberg-Marquardt style.
- The cross second derivative of `MarkovMatrix` with a non-uniform `skew`
  was wrong.

### Performance
- `MarkovMatrix` and `MarkovVector` are vectorised (2x to 4x faster).
- `MaxLike.graph()` and `Func.graph()` build the computation graph of a
  model. Export it as Graphviz DOT or Mermaid text, or render it with the
  optional extra `pip install "maxlike[graph]"`.

### Performance
- Likelihood intermediates are cached within a Newton step, and tensor
  contractions use einsum. Large models run 1.3x to 2.9x faster; results
  are unchanged.

## 2.4.0

### Compatibility
- Works with current Python (3.10–3.13), NumPy (including 2.x) and SciPy.
- Python 3.9 and older are no longer supported.
- Minimum versions: NumPy 1.21, SciPy 1.7 and pandas 1.3.
- The Gaussian copula now uses a closed-form bivariate normal CDF based on
  Owen's T function, replacing `scipy.stats.mvn.mvnun`, which was removed
  from SciPy.
- The `six` dependency was removed.

### Fixes
- AIC/BIC no longer count fixed parameters as degrees of freedom.
- `std_error()` works for models without constraints.
- `fit(method="broyden", verbose=...)` no longer raises `NameError`.
- A successful Newton fit now sets `converged = True`.
- `akaike_information_criterion` is the new name of the AIC method;
  `akaine_information_criterion` is kept as an alias.

### Packaging and CI
- Licensed under the GNU GPL v3 or later.
- Packaging moved to `pyproject.toml`.
- CI moved from Travis to GitHub Actions; releases are published to PyPI
  from GitHub.
