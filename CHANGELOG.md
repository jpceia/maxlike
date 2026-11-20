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

### Fixed
- The convergence test no longer divides by the objective; refitting from
  the optimum no longer raises a ConvergenceError.
- Newton steps that would decrease the objective (Hessian not negative
  definite) are damped Levenberg-Marquardt style.
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
