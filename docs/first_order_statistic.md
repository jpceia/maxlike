# `FirstOrderStatistic`: who wins, and with what value

`maxlike.func.FirstOrderStatistic` models observations in which several
participants each draw a latent value, the **smallest value wins**, and only
the **winner** and the **winning value** are observed. The values of the
other participants are never seen; we only know that they were larger.

Examples:

| Domain | Participants | Latent value | Observed |
|---|---|---|---|
| Public procurement (lowest bid wins) | firms bidding for a tender | log bid / base price | winning firm, winning price |
| Races | runners, drivers | finishing time | winner, winning time |
| Competing risks | possible causes | time to the event | first cause, time of the first event |

For "largest value wins" (e.g. highest-bid auctions), pass the negated
values.

The implementation is in [`maxlike/func/order_statistic.py`](../maxlike/func/order_statistic.py);
a worked example on synthetic procurement data is in
[`docs/examples/procurement_auction.py`](examples/procurement_auction.py).

## 1. Model of the latent values

For observation $t$ (a tender) and participant $j$ (a firm), the latent value
is

$$
v_{tj} = c_t + \mu_j + \sigma\,\varepsilon_{tj},
\qquad \varepsilon_{tj} \sim \mathcal N(0, 1) \text{ i.i.d.}
$$

- $\mu_j$: **location of participant $j$**, e.g. the log markup of a firm
  (lower = more aggressive).
- $\sigma$: **dispersion** of each participant's values around its location
  (parametrised as $\log\sigma$).
- $c_t$: **common factor of observation $t$**, shared by all participants,
  e.g. the cost of executing the tender. A costly tender raises every bid
  together, so the values of an observation are **not independent**; they
  are independent **given** $c_t$.

Two specifications of the common factor:

| | Common factor | Parameters |
|---|---|---|
| **A. independent given covariates** (`common_factor=False`) | $c_t = x_t^\top\beta$ | $\mu, \log\sigma, \beta$ |
| **B. latent common factor** (`common_factor=True`) | $c_t = x_t^\top\beta + \tau\,u_t,\ u_t \sim \mathcal N(0,1)$ | $\mu, \log\sigma, \beta, \tau$ |

$x_t$ are covariates of the observation (include a column of ones for an
intercept). $\tau$ measures how much the values of an observation move
together **beyond** what the covariates explain. Model A is model B with
$\tau = 0$.

## 2. Likelihood of one observation

Write the standardised distance between the winning value $v_t$ and
participant $j$'s distribution, given the common factor $c$:

$$
z_{tj}(c) = \frac{v_t - c - \mu_j}{\sigma}.
$$

Given $c$, the values are independent. Participant $w$ wins with value $v_t$
when its value is $v_t$ **and** every other participant $j \in P_t$ drew more:

$$
L_t(c) \;=\; \underbrace{\frac{1}{\sigma}\,\varphi\!\big(z_{tw}(c)\big)}_{\text{density of the winner's value}}
\;\prod_{j \in P_t,\ j \neq w} \underbrace{\Phi\!\big(-z_{tj}(c)\big)}_{\Pr(v_{tj} > v_t)}
$$

with $\varphi$, $\Phi$ the standard normal density and distribution
function, and $P_t$ the set of participants of observation $t$. In logs:

$$
\ell_t(c) = -\log\sigma + \log\varphi(z_{tw}) + \sum_{j \in P_t \setminus w} \log\Phi(-z_{tj}).
$$

The participants that lost contribute only through **survival
probabilities**: this is what lets the model learn about firms that rarely
win.

**Model A** uses $\ell_t = \ell_t(x_t^\top\beta)$ directly.

## 3. Integrating the common factor (model B)

In model B, $u_t$ is never observed, so it is integrated out:

$$
L_t = \int L_t\big(x_t^\top\beta + \tau u\big)\,\varphi(u)\,du
\;\approx\; \sum_{k=1}^{K} \omega_k\, L_t\big(x_t^\top\beta + \tau u_k\big),
$$

with Gauss–Hermite nodes $u_k$ and weights $\omega_k$ for the standard
normal (`numpy.polynomial.hermite_e.hermegauss`, normalised to sum to one;
$K = 15$ by default, which gives the same estimates as $K = 41$ in the
validation below). It is computed stably on the log scale:

$$
\log L_t = \operatorname{logsumexp}_k\big(\log\omega_k + \ell_{tk}\big),
\qquad \ell_{tk} = \ell_t\big(x_t^\top\beta + \tau u_k\big).
$$

`FirstOrderStatistic` returns $\log L_t$ for every observation, and
`maxlike.LogLikelihood` maximises $\sum_t N_t \log L_t$ (with weights
$N_t$, usually ones).

## 4. Derivatives

maxlike uses Newton's method with **exact** first and second derivatives;
they are derived analytically below and checked against finite differences
in `tests/order_statistic_test.py`.

### Per participant, in $z$

For each term of $\ell$, its first and second derivatives with respect to
$z$:

| | term | $g = \partial/\partial z$ | $h = \partial^2/\partial z^2$ |
|---|---|---|---|
| winner | $\log\varphi(z)$ | $-z$ | $-1$ |
| other participant | $\log\Phi(-z)$ | $-\lambda(z)$ | $-\lambda(z)\,\big(\lambda(z) - z\big)$ |
| non-participant | $0$ | $0$ | $0$ |

where $\lambda(z) = \varphi(z)/\Phi(-z)$ is the normal hazard (inverse Mills
ratio), computed as $\exp\!\big(\log\varphi(z) - \log\Phi(-z)\big)$ for
stability.

### Chain rule through $z$

With $s = \log\sigma$, $z_{tj} = (v_t - x_t^\top\beta - \tau u_k - \mu_j)\,e^{-s}$, so

$$
\frac{\partial z_j}{\partial \mu_m} = -\frac{\delta_{jm}}{\sigma},\quad
\frac{\partial z_j}{\partial s} = -z_j,\quad
\frac{\partial z_j}{\partial \beta} = -\frac{x_t}{\sigma},\quad
\frac{\partial z_j}{\partial \tau} = -\frac{u_k}{\sigma},
$$

and the only non-zero second derivatives of $z$ are

$$
\frac{\partial^2 z_j}{\partial s^2} = z_j,\quad
\frac{\partial^2 z_j}{\partial s\,\partial \mu_j} = \frac{1}{\sigma},\quad
\frac{\partial^2 z_j}{\partial s\,\partial \beta} = \frac{x_t}{\sigma},\quad
\frac{\partial^2 z_j}{\partial s\,\partial \tau} = \frac{u_k}{\sigma}.
$$

Then, for each observation $t$ and node $k$,
$\nabla\ell = \sum_j g_j \nabla z_j - e_s$ and
$\nabla^2\ell = \sum_j \big(h_j \nabla z_j \nabla z_j^\top + g_j \nabla^2 z_j\big)$.
With the sums $G = \sum_j g_j$, $H = \sum_j h_j$,
$A = \sum_j (h_j z_j + g_j)$ and $B = \sum_j (h_j z_j^2 + g_j z_j)$:

| Gradient | |
|---|---|
| $\partial\ell/\partial\mu_m$ | $-g_m/\sigma$ |
| $\partial\ell/\partial s$ | $-\sum_j g_j z_j - 1$ |
| $\partial\ell/\partial\beta$ | $-G\,x_t/\sigma$ |
| $\partial\ell/\partial\tau$ | $-G\,u_k/\sigma$ |

| Hessian block | |
|---|---|
| $\mu_m\mu_n$ | $\delta_{mn}\,h_m/\sigma^2$ (diagonal) |
| $s\,\mu_m$ | $(h_m z_m + g_m)/\sigma$ |
| $s\,s$ | $B$ |
| $\beta\,\mu_m$ | $h_m\,x_t/\sigma^2$ |
| $\beta\,s$ | $A\,x_t/\sigma$ |
| $\beta\beta^\top$ | $H\,x_t x_t^\top/\sigma^2$ |
| $\tau\,\mu_m$ | $h_m u_k/\sigma^2$ |
| $\tau\,s$ | $A\,u_k/\sigma$ |
| $\tau\,\beta$ | $H\,x_t u_k/\sigma^2$ |
| $\tau\,\tau$ | $H\,u_k^2/\sigma^2$ |

### Through the mixture over nodes

With the posterior weights of the nodes
$\pi_{tk} = \omega_k e^{\ell_{tk}} / L_t$ (the softmax of
$\log\omega_k + \ell_{tk}$):

$$
\nabla \log L_t = \sum_k \pi_{tk}\,\nabla\ell_{tk},
\qquad
\nabla^2 \log L_t = \sum_k \pi_{tk}\big(\nabla^2\ell_{tk} + \nabla\ell_{tk}\nabla\ell_{tk}^\top\big) - \nabla\log L_t\,\nabla\log L_t^\top.
$$

In model A there is a single node with weight one, and these reduce to
$\nabla\ell_t$ and $\nabla^2\ell_t$. Note that the mixture couples all the
participants of an observation: the $\mu_m\mu_n$ block of
$\nabla^2\log L_t$ is no longer diagonal.

## 5. Identification

- **Overall level.** Adding a constant to every $\mu_j$ and subtracting it
  from the intercept leaves every $z$ unchanged. Fix it with a constraint,
  e.g. $\sum_j \mu_j = 0$ (`mle.add_constraint(0, Linear([1]))`). Without it
  `fit` raises `IdentifiabilityError`: *"flat along all of 'mu' (+1) +
  'beta'[0] (-1)"*.
- **Who wins** depends only on the differences $\mu_i - \mu_j$ (in units of
  $\sigma$): the common factor shifts everybody.
- **The winning value** is the common factor plus the minimum of the
  participants' values. $\tau$ (common) and $\sigma$ (individual) are told
  apart by how the spread of winning values changes with the **number of
  participants** (the minimum of more draws is less dispersed, the common
  factor is not) and by the normal shape. Observations with varying numbers
  of participants help; so do the losing values when they are published,
  since the common factor cancels in their differences.
- **Why not a free $c_t$ per observation?** With a single (winner, value)
  per observation, a free $c_t$ would absorb the winning value exactly
  (incidental parameters). The common factor is therefore a random effect,
  integrated out, with covariates for its predictable part.

## 6. Usage

```python
import numpy as np
import maxlike
from maxlike.func import FirstOrderStatistic, Linear

# values: (T,) winning log price relative to the base price
# winner: (T,) index of the winning firm
# participants: (T, J) True for the firms that bid
# X: (T, Q) covariates, first column of ones
mle = maxlike.LogLikelihood()
mle.model = FirstOrderStatistic(values, winner, participants, X,
                                common_factor=True)
mle.add_constraint(0, Linear([1]))          # sum(mu) = 0
mle.add_param(np.zeros(J), name="mu", labels=[firm_names])
mle.add_param(np.log(.2), name="log_sigma")
mle.add_param(np.r_[values.mean(), np.zeros(Q - 1)], name="beta")
mle.add_param(.05, name="tau")             # start away from 0
mle.fit(N=np.ones(len(values)))
mle.summary()
```

## 7. Validation on synthetic data

From `docs/examples/procurement_auction.py`: 15 firms, 3000 tenders with 2
to 6 bidders each, only the winner and the winning price observed; true
$\sigma = 0.10$, intercept $-0.15$.

| True data | Model | $\mu$: RMSE vs truth | $\mu$ within 2 SE | $\hat\sigma$ | intercept | $\hat\tau$ | hold-out log-lik. per tender |
|---|---|---|---|---|---|---|---|
| common factor, $\tau = 0.12$ | A. independent | 0.037 | 47% | 0.192 | −0.031 | — | −0.801 |
| | **B. common factor** | **0.009** | **93%** | **0.109** | **−0.136** | **0.114** | **−0.759** |
| no common factor, $\tau = 0$ | A. independent | 0.006 | 100% | 0.100 | −0.152 | — | −0.053 |
| | B. common factor | 0.006 | 100% | 0.100 | −0.152 | 0.000 | −0.053 |

Ignoring a common factor that exists (model A on the first data set)
misattributes the shared variation to the firms: the dispersion is almost
doubled, the firm markups are biased and their standard errors are too
small. Model B recovers the truth, and when there is no common factor it
reduces to model A. Across four further data sets the estimates of model B
were within −0.7 to +1.2 standard errors of the truth.

## 8. Limitations

- Normal shape for the individual values and the common factor; a single
  dispersion $\sigma$ shared by all participants.
- The participants of each observation must be known. If only their number
  is known, an anonymous "fringe" participant type would be needed.
- Dense participants × participants Hessian blocks per observation: fine
  for tens to a few hundred participants.
- Participation is taken as given: firms that choose to bid when their
  markup is low (endogenous entry) or collusion break conditional
  independence.
