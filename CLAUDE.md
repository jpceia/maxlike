# MaxLike

A library for fitting statistical models with Newton's method, using exact
Hessians that keep their sparsity structure. See README.md for the user-facing
description, and the "How it works" section there for the architecture.

## Commands

```bash
pip install -e ".[test]"            # library + pytest
python -m pytest tests              # full suite, ~25 s
python -m pytest tests/derivatives_test.py -q
pip install -e ".[notebooks]"       # adds matplotlib, jupyter, graphviz (Python)
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/<name>.ipynb
python docs/examples/<name>.py
```

CI (`.github/workflows/tests.yml`) runs:
- the tests on Python 3.10–3.13;
- the tests with the **minimum dependencies**: numpy 1.21, scipy 1.7 and
  pandas 1.3. Do not use newer APIs without a fallback;
- every notebook, executed with Graphviz installed.

## Layout

- `maxlike/maxlike_base.py`: `MaxLike`, which holds the parameters
  (`Param`), constraints, regularizations, the Newton/Broyden solver and the
  dense or sparse assembly of the KKT system.
- `maxlike/maxlike.py`: the scores (likelihoods): `Poisson`, `Logistic`,
  `Finite`, `NegativeBinomial`, `LogLikelihood`, ...
- `maxlike/func/`: the model building blocks (`Func` subclasses):
  - `func.py`: elementary functions, `Take`, and the priors
    `QuadraticForm`, `Quadratic` and `RandomWalk`;
  - `func_op.py`: `Sum`, `Product`, `Collapse`;
  - `func_base.py`: `Func`, `FuncMeta`, `Affine`, `Compose`;
  - plus `func_proba.py`, `matrix.py`, `copula.py`, `tennis.py` and
    `order_statistic.py`.
- `maxlike/tensor/`: `Tensor`, `GenericTensor` and `IndexedTensor`, the
  structured derivatives. The axis groups are `x_dim`, `y_dim`, `n` (features)
  and `dim` (distribution). `x_map`/`y_map` mark diagonal axes, which are
  stored by their diagonal only.
- `maxlike/linalg.py`: sparse direct solver (Schur elimination of diagonal
  blocks).
- `maxlike/inference.py`: `summary`, `predict`, `simulate` and labels.
- `maxlike/graph.py`: model graphs (DOT and Mermaid).
- `docs/examples/*.py`: runnable examples. `docs/notebooks/*.ipynb` has one
  notebook per example, which reuses the example's functions.

## Writing a Func

- Define `__call__(params)`, `grad(params, i)` and `hess(params, i, j)` for
  `j <= i`, all returning `Tensor`s. `eval` is optional.
- `FuncMeta` wraps these methods: `grad(params)` and `hess(params)` with no
  index return the list of all gradients, and the lower-triangular list of
  lists of all second derivatives. A subclass of a concrete Func inherits the
  methods it does not define.
- Keep the structure. Return diagonal derivatives with `grad_tensor(...,
  True)` / `hess_tensor(..., True, True)`, and use `null_func` for zero
  derivatives. Never densify where a map or an `IndexedTensor` can be used.
- Scalar functions used as constraints or regularizations can also define
  `hess_sparse(params, i, j)`, returning a scipy.sparse block over the
  flattened parameters (or `None` when the block is zero). It is used when
  the KKT system is assembled sparse, which happens from 500 free
  parameters on with observation-indexed models.
- Every new Func gets a case in `tests/derivatives_test.py` (finite
  differences, through both `eval` and `grad`/`hess`). If it supports sparse
  assembly, add a dense-versus-sparse fit check in
  `tests/indexed_model_test.py` (`both_assemblies`).

## Conventions

- Code, comments, docstrings and docs are in English. Docstrings state the
  formula first (`Name(args)(params) = ...`), then the meaning, then
  `Parameters`.
- A new model or analysis comes with:
  - a script in `docs/examples/`, with a `main()` that returns the fitted
    objects;
  - a case in `tests/examples_test.py`;
  - a notebook in `docs/notebooks/`, committed already executed, using the
    chart style in `plot_style.py`;
  - rows in the example table of `README.md` and in
    `docs/notebooks/README.md`.
- Use synthetic data with a known truth when `tests/data` lacks what a model
  needs, as `race.py` and `football_dynamic.py` do. `tests/data/data2.csv`
  holds two seasons of 18 teams in chronological order, but has no dates.
- Record user-visible changes under "Unreleased" in `CHANGELOG.md` (Added /
  Changed / Fixed / Performance).
- Performance is a feature. Before changing a hot path (Tensor ops, Hessian
  assembly, priors), time a grid fit and an indexed fit, and compare the
  results to many digits.
