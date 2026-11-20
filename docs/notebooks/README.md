# Notebooks

One notebook per example, each with the theory of the model (with the
formulas), the data, the computation graph of the model, the fit with its
standard errors, predictions, and checks of the fit. They are committed
already executed, so GitHub shows the charts without running anything.

| Notebook | Model | Checks |
|---|---|---|
| [Football goals](football_goals.ipynb) | Poisson goals, attack and defence per team, home advantage | goal distribution, calibration of the scoring rates, dispersion |
| [Football scores](football_scores.ipynb) | joint score distribution (two censored Poisson distributions), 1X2 and goal-line markets | observed against fitted scores, reliability of the 1X2 probabilities |
| [Market calibration](football_market_calibration.ipynb) | the score model fitted to market 1X2 probabilities (Kullback–Leibler) | market against model probabilities, censored against truncated goal distributions |
| [Dynamic strengths](football_dynamic.ipynb) | attack and defence per team and round, tied by a Gaussian random walk; step size chosen by one-round-ahead forecasts | estimated paths against a known truth, error per round against the static model, forecast score against the step size |
| [Tennis](tennis.ipynb) | point → game / tie-break → set recursions, one strength per player | calibration of set-win probabilities, distribution of set scores |
| [Tennis with serve](tennis_serve.ipynb) | the tennis model with one global serve advantage (`ServeSet`: serve alternating, 50/50 on who serves first) | likelihood ratio against the base model, set-score shape, calibration |
| [Polymarket: blockade and ceasefire](polymarket_iran.ipynb) | two ladders of prediction-market contracts as hazard curves: a latent factor (state-space model, from the price history) and a four-state Markov chain | correlation of the two hazards as a parameter, curves without the noise, mean reversion, prediction from the other curve |
| [Polymarket: fair mid of related markets](polymarket_fair_mid.ipynb) | generic model for K ladders (blockade, ceasefire, Strait of Hormuz) on hourly prices: level and front factors per hazard curve, joint random walk, (extended) Kalman filter with exact scores | correlations within and across curves, fair mid against bid–ask, prediction at 1 and 3 hours (in and out of sample) |
| [Races](race.ipynb) | only the winner and the winning time are observed (`FirstOrderStatistic`), with and without a race-day factor | estimates against a known truth, held-out win probabilities |

They reuse the functions of the scripts in [`docs/examples`](../examples);
`plot_style.py` holds the chart style they share.

To run them:

```bash
pip install -e ".[notebooks]"
jupyter lab docs/notebooks
```

The model graphs need the Graphviz program (`dot`) as well; without it they
are shown as text.
