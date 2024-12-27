---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: mclmc
  language: python
  name: python3
---

# Microcanonical Langevin Monte Carlo

This is an algorithm based on https://arxiv.org/abs/2212.08549 ({cite:p}`robnik2023microcanonical`, {cite:p}`robnik2023microcanonical2`). A website with detailed information can be found [here](https://microcanonical-monte-carlo.netlify.app/). The algorithm is provided in both adjusted (i.e. with an Metropolis-Hastings step) and unadjusted versions; by default we use "MCLMC" to refer to the unadjusted version.

The original derivation comes from thinking about the microcanonical ensemble (a concept from statistical mechanics), but the upshot is that we integrate the following SDE:

$$
\frac{d}{dt}\begin{bmatrix}
x \\
u
\end{bmatrix}
=
\begin{bmatrix}
u \\
-P(u)(\nabla S(x)/(d − 1)) + \eta P(u)dW
\end{bmatrix}
$$

where $u$ is an auxilliary variable, $S(x)$ is the negative log PDF of the distribution from which we are sampling and the last term describes spherically symmetric noise. After $u$ is marginalized out, this converges to the target PDF, $p(x) \propto e^{-S(x)}$.

## How to run MCLMC in BlackJax

It is very important to use the tuning algorithm provided, which controls the step size of the integrator and also $L$, a parameter related to $\eta$ above.

An example is given below, of a 1000 dim Gaussian (of which 2 dimensions are plotted).

```{code-cell} ipython3
:tags: [hide-cell]

import matplotlib.pyplot as plt

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 19
```

```{code-cell} ipython3
:tags: [remove-output]

import jax

from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{code-cell} ipython3
import blackjax
from blackjax.mcmc.adjusted_mclmc import rescale
import numpy as np
import jax.numpy as jnp
```

```{code-cell} ipython3
def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    # find values for L and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples
```

```{code-cell} ipython3
# run the algorithm on a high dimensional gaussian, and show two of the dimensions

sample_key, rng_key = jax.random.split(rng_key)
samples = run_mclmc(
    logdensity_fn=lambda x: -0.5 * jnp.sum(jnp.square(x)),
    num_steps=1000,
    initial_position=jnp.ones((1000,)),
    key=sample_key,
    transform=lambda state, _: state.position[:2],
)
samples.mean()
```

```{code-cell} ipython3
plt.scatter(x=samples[:, 0], y=samples[:, 1], alpha=0.1)
plt.axis("equal")
plt.title("Scatter Plot of Samples")
```

# Second example: Stochastic Volatility

This is ported from Jakob Robnik's [example notebook](https://github.com/JakobRobnik/MicroCanonicalHMC/blob/master/notebooks/tutorials/advanced_tutorial.ipynb)

```{code-cell} ipython3
import matplotlib.dates as mdates

from numpyro.examples.datasets import SP500, load_dataset
from numpyro.distributions import StudentT

# get the data
_, fetch = load_dataset(SP500, shuffle=False)
SP500_dates, SP500_returns = fetch()


# figure setup
_, ax = plt.subplots(figsize=(12, 5))
ax.spines["right"].set_visible(False)  # remove the upper and the right axis lines
ax.spines["top"].set_visible(False)

ax.xaxis.set_major_locator(mdates.YearLocator())  # dates on the xaxis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

# plot data
dates = mdates.num2date(mdates.datestr2num(SP500_dates))
ax.plot(dates, SP500_returns, ".", markersize=3, color="steelblue")
ax.set_xlabel("time")
ax.set_ylabel("S&P500 returns")
```

```{code-cell} ipython3
dim = 2429

lambda_sigma, lambda_nu = 50, 0.1


def logp_volatility(x):
    """log p of the target distribution"""

    sigma = (
        jnp.exp(x[-2]) / lambda_sigma
    )  # we used log-transformation to make x unconstrained
    nu = jnp.exp(x[-1]) / lambda_nu

    prior2 = (jnp.exp(x[-2]) - x[-2]) + (
        jnp.exp(x[-1]) - x[-1]
    )  # - log prior(sigma, nu)
    prior1 = (dim - 2) * jnp.log(sigma) + 0.5 * (
        jnp.square(x[0]) + jnp.sum(jnp.square(x[1:-2] - x[:-3]))
    ) / jnp.square(
        sigma
    )  # - log prior(R)
    lik = -jnp.sum(
        StudentT(df=nu, scale=jnp.exp(x[:-2])).log_prob(SP500_returns)
    )  # - log likelihood

    return -(lik + prior1 + prior2)


def transform(x):
    """transform x back to the parameters R, sigma and nu (taking the exponent)"""

    Rn = jnp.exp(x[:-2])
    sigma = jnp.exp(x[-2]) / lambda_sigma
    nu = jnp.exp(x[-1]) / lambda_nu

    return jnp.concatenate((Rn, jnp.array([sigma, nu])))


def prior_draw(key):
    """draws x from the prior"""

    key_walk, key_exp1, key_exp2 = jax.random.split(key, 3)

    sigma = (
        jax.random.exponential(key_exp1) / lambda_sigma
    )  # sigma is drawn from the exponential distribution

    def step(track, useless):  # one step of the gaussian random walk
        randkey, subkey = jax.random.split(track[1])
        x = (
            jax.random.normal(subkey, shape=track[0].shape, dtype=track[0].dtype)
            + track[0]
        )
        return (x, randkey), x

    x = jnp.empty(dim)
    x = x.at[:-2].set(
        jax.lax.scan(step, init=(0.0, key_walk), xs=None, length=dim - 2)[1] * sigma
    )  # = log R_n are drawn as a Gaussian random walk realization
    x = x.at[-2].set(
        jnp.log(sigma * lambda_sigma)
    )  # sigma ~ exponential distribution(lambda_sigma)
    x = x.at[-1].set(
        jnp.log(jax.random.exponential(key_exp2))
    )  # nu ~ exponential distribution(lambda_nu)

    return x
```

```{code-cell} ipython3
key1, key2, rng_key = jax.random.split(rng_key, 3)
samples = run_mclmc(
    logdensity_fn=logp_volatility,
    num_steps=10000,
    initial_position=prior_draw(key1),
    key=key2,
    transform=lambda x: x,
)

samples = transform(samples.position)
```

```{code-cell} ipython3
R = np.array(samples)[:, :-2]  # remove sigma and nu parameters
R = np.sort(R, axis=0)  # sort samples for each R_n
num_samples = len(R)
lower_quartile, median, upper_quartile = (
    R[num_samples // 4, :],
    R[num_samples // 2, :],
    R[3 * num_samples // 4, :],
)

# figure setup
_, ax = plt.subplots(figsize=(12, 5))
ax.spines["right"].set_visible(False)  # remove the upper and the right axis lines
ax.spines["top"].set_visible(False)

ax.xaxis.set_major_locator(mdates.YearLocator())  # dates on the xaxis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

# plot data
ax.plot(dates, SP500_returns, ".", markersize=3, color="steelblue")
ax.plot(
    [], [], ".", markersize=10, color="steelblue", alpha=0.5, label="data"
)  # larger markersize for the legend
ax.set_xlabel("time")
ax.set_ylabel("S&P500 returns")

# plot posterior
ax.plot(dates, median, color="navy", label="volatility posterior")
ax.fill_between(dates, lower_quartile, upper_quartile, color="navy", alpha=0.5)

ax.legend()
```

## Adjusted MCLMC

Blackjax also provides a version of the algorithm with MH adjustment, along with its own tuning algorithm. This has the same two hyperparameters, `step_size` and `L`. Here `L`, which represents the decoherence length, determines the length of the proposal, since the momentum is refreshed (and therefore decorrelated) after each proposal.

There are two versions of the algorithm. In the first, wthe proposed trajectory has no Langevin noise in it, but the trajectory length is randomized. In the second, the trajectory length is fixed, but there is Langevin noise. Here is how to run both:

```{code-cell} ipython3
def run_adjusted_mclmc(
    logdensity_fn,
    num_steps,
    initial_position,
    key,
    diagonal_preconditioning=False,
    transform = lambda state, _: state.position,
    random_trajectory_length=True,
    L_proposal_factor = jnp.inf
):

    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        sqrt_diag_cov=sqrt_diag_cov,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        L_proposal_factor=L_proposal_factor,
    )

    target_acc_rate = 0.9 # our recommendation

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.0, # our recommendation
        diagonal_preconditioning=diagonal_preconditioning,
    )

    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L

    alg = blackjax.adjusted_mclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
    )

    _, out = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )

    return out
```

```{code-cell} ipython3
# run the algorithm on a high dimensional gaussian, and show two of the dimensions

sample_key, rng_key = jax.random.split(rng_key)
samples = run_adjusted_mclmc(
    logdensity_fn=lambda x: -0.5 * jnp.sum(jnp.square(x)),
    num_steps=2000,
    initial_position=jnp.ones((1000,)),
    key=sample_key,
)
plt.scatter(x=samples[:, 0], y=samples[:, 1], alpha=0.1)
plt.axis("equal")
plt.title("Scatter Plot of Samples")
```

```{code-cell} ipython3
# run the algorithm on a high dimensional gaussian, and show two of the dimensions

sample_key, rng_key = jax.random.split(rng_key)
samples = run_adjusted_mclmc(
    logdensity_fn=lambda x: -0.5 * jnp.sum(jnp.square(x)),
    num_steps=2000,
    initial_position=jnp.ones((1000,)),
    key=sample_key,
    random_trajectory_length=False,
    L_proposal_factor=1.25
)
plt.scatter(x=samples[:, 0], y=samples[:, 1], alpha=0.1)
plt.axis("equal")
plt.title("Scatter Plot of Samples")
```
