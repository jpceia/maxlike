# Changelog

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
- Packaging moved to `pyproject.toml`.
- CI moved from Travis to GitHub Actions; releases are published to PyPI
  from GitHub.
