---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Sparse regression

In this example we will use a sparse binary regression with hierarchies on the scale of the independent variable’s parameters that function as a proxy for variable selection. We will use the Horseshoe prior to {cite:p}`carvalho2010horseshoe` to ensure sparsity.

The Horseshoe prior consists in putting a prior on the scale of the regression parameter $\beta$: the product of a global $\tau$ and local $\lambda$ parameter that are both concentrated at $0$, thus allowing the corresponding regression parameter to degenerate at $0$ and effectively excluding this parameter from the model. This kind of model is challenging for samplers: the prior on $\beta$'s scale parameter creates funnel geometries that are hard to efficiently explore {cite:p}`papaspiliopoulos2007general`.

Mathematically, we will consider the following model:

\begin{align*}
\tau &\sim \operatorname{C}^+(0, 1)\\
\boldsymbol{\lambda} &\sim \operatorname{C}^+(0, 1)\\
\boldsymbol{\beta} &\sim \operatorname{Normal}(0, \tau \lambda)\\
\\
p &= \operatorname{sigmoid}\left(- X.\boldsymbol{\beta}\right)\\
y &\sim \operatorname{Bernoulli}(p)\\
\end{align*}

The model is run on its *non-centered parametrization* {cite:p}`papaspiliopoulos2007general` with data from the numerical version of the German credit dataset. The target posterior is defined by its likelihood. We implement the model using [Aesara](https://github.com/aesara-devs/aesara):

```{code-cell} ipython3
:tags: [hide-cell]

import matplotlib.pyplot as plt

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
```

```{code-cell} ipython3
:tags: [remove-output]

import jax

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{code-cell} ipython3
import aesara.tensor as at

X_at = at.matrix('X')

srng = at.random.RandomStream(0)

tau_rv = srng.halfcauchy(0, 1)
lambda_rv = srng.halfcauchy(0, 1, size=X_at.shape[-1])

sigma = tau_rv * lambda_rv
beta_rv = srng.normal(0, sigma, size=X_at.shape[-1])

eta = X_at @ beta_rv
p = at.sigmoid(-eta)
Y_rv = srng.bernoulli(p, name="Y")
```

```{note}
The non-centered parametrization is not necessarily adapted to every geometry. One should always check *a posteriori* the sampler did not encounter any funnel geomtry.
```

## German credit dataset

We will use the sparse regression model on the German credit dataset {cite:p}`dua2017machine`. We use the numeric version that is adapted to models that cannot handle categorical data:

```{code-cell} ipython3
import pandas as pd

data = pd.read_table(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
  header=None,
  delim_whitespace=True
)
```

Each row in the dataset corresponds to a different customer. The dependent variable $y$ is equal to $1$ when the customer has good credit and $2$ when it has bad credit; we encode it so a customer with good credit corresponds to $1$, a customer with bad credit $1$:

```{code-cell} ipython3
y = -1 * (data.iloc[:, -1].values - 2)
```

```{code-cell} ipython3
r_bad = len(y[y==0.]) / len(y)
r_good = len(y[y>1]) /  len(y)

print(f"{r_bad*100}% of the customers in the dataset are classified as having bad credit.")
```

The regressors are defined on different scales so we normalize their values, and add a column of $1$ that corresponds to the intercept:

```{code-cell} ipython3
import numpy as np

X = (
    data.iloc[:, :-1]
    .apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0)
    .values
)
X = np.concatenate([np.ones((1000, 1)), X], axis=1)
```

## Models

We generate a function that computes the model's logdensity using [AePPL](https://github.com/aesara-devs/aeppl). We transform the values of $\tau$ and $\lambda$ so the sampler can operate on variables defined on the real line:

```{code-cell} ipython3
import aesara
import aeppl
from aeppl.transforms import TransformValuesRewrite, LogTransform

transforms_op = TransformValuesRewrite(
     {lambda_rv: LogTransform(), tau_rv: LogTransform()}
)

logdensity, value_variables = aeppl.joint_logprob(
    tau_rv,
    lambda_rv,
    beta_rv,
    realized={Y_rv: at.as_tensor(y)},
    extra_rewrites=transforms_op
)


logdensity_aesara_fn = aesara.function([X_at] + list(value_variables), logdensity, mode="JAX")

def logdensity_fn(x):
    tau = x['log_tau']
    lmbda = x['log_lmbda']
    beta = x['beta']
    return logdensity_aesara_fn.vm.jit_fn(X, tau, lmbda, beta)[0]
```

Let us now define a utility function that builds a sampling loop:

```{code-cell} ipython3
def inference_loop(rng_key, init_state, kernel, n_iter):
    keys = jax.random.split(rng_key, n_iter)

    def step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info
```

### MEADS

The MEADS algorithm {cite:p}`hoffman2022tuning` is a combination of Generalized HMC with a parameter tuning procedure. Let us initialize the position of the chain first:

```{code-cell} ipython3
num_chains = 128
num_warmup = 2000
num_samples = 2000

rng_key, key_b, key_l, key_t = jax.random.split(rng_key, 4)
init_position = {
    "beta": jax.random.normal(key_b, (num_chains, X.shape[1])),
    "log_lmbda": jax.random.normal(key_l, (num_chains, X.shape[1])),
    "log_tau": jax.random.normal(key_t, (num_chains,)),
}
```

Here we will not use the adaptive version of the MEADS algorithm, but instead use their heuristics as an adaptation procedure for Generalized Hamiltonian Monte Carlo kernels:

```{code-cell} ipython3
import blackjax

rng_key, key_warmup, key_sample = jax.random.split(rng_key, 3)
meads = blackjax.meads_adaptation(logdensity_fn, num_chains)
(state, parameters), _ = meads.run(key_warmup, init_position, num_warmup)
kernel = blackjax.ghmc(logdensity_fn, **parameters).step

# Choose the last state of the first k chains as a starting point for the sampler
n_parallel_chains = 4
init_states = jax.tree_util.tree_map(lambda x: x[:n_parallel_chains], state)
keys = jax.random.split(key_sample, n_parallel_chains)
samples, info = jax.vmap(inference_loop, in_axes=(0, 0, None, None))(
    keys, init_states, kernel, num_samples
    )
```

Let us look a high-level summary statistics for the inference, including the split-Rhat value and the number of effective samples:

```{code-cell} ipython3
from numpyro.diagnostics import print_summary

print_summary(samples.position)
```

Let's check if there are any divergent transitions

```{code-cell} ipython3
np.sum(info.is_divergent, axis=1)
```

We warned earlier that the non-centered parametrization was not a one-size-fits-all solution to the funnel geometries that can be present in the posterior distribution. Although there was no divergence, it is still worth checking the posterior interactions between the coefficients to make sure the posterior geometry did not get in the way of sampling:

```{code-cell} ipython3
n_pred = X.shape[-1]
n_col = 4
n_row = (n_pred + n_col - 1) // n_col

_, axes = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 2))
axes = axes.flatten()
for i in range(n_pred):
    ax = axes[i]
    ax.plot(samples.position["log_lmbda"][...,i], 
            samples.position["beta"][...,i], 
            'o', ms=.4, alpha=.75)
    ax.set(
        xlabel=rf"$\lambda$[{i}]",
        ylabel=rf"$\beta$[{i}]",
    )
for j in range(i+1, n_col*n_row):
    axes[j].remove()
plt.tight_layout();
```

While some parameters (for instance the 15th) exhibit no particular correlations, the funnel geometry can still be observed for a few of them (4th, 13th, etc.). Ideally one would adopt a centered parametrization for those parameters to get a better approximation to the true posterior distribution, but here we also assess the ability of the sampler to explore these funnel geometries.

+++

We can convince ourselves that the Horseshoe prior induces sparsity on the regression coefficients by looking at their posterior distribution:

```{code-cell} ipython3
_, axes = plt.subplots(n_row, n_col, sharex=True, figsize=(n_col * 3, n_row * 2))
axes = axes.flatten()
for i in range(n_pred):
    ax = axes[i]
    ax.hist(samples.position["beta"][..., i],
            bins=50, density=True, histtype="step")
    ax.set_xlabel(rf"$\beta$[{i}]")
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
ax.set_xlim([-2, 2])
for j in range(i+1, n_col*n_row):
    axes[j].remove()
plt.tight_layout();
```

Indeed, many of the parameters are centered around $0$.

```{note}
It is interesting to notice that the interactions for the parameters with large values do not exhibit funnel geometries.
```

+++

## Bibliography

```{bibliography}
:filter: docname in docnames
```
