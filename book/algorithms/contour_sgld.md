---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Contour stochastic gradient Langevin dynamics
#### An elegant sampler for simulations of multi-modal distributions

+++

Sampling multi-modal distributions in big data is quite challenging due to the higher energy barriers. It is often empirically solved via cyclical learning rates or different initializations (parallel chains).

To solve this issue, Contour SgLD resorts to the importance sampling approach, which adatively simulates from a flat landscape and then recovers the landscape through importance weights. In this notebook we will compare SGLD and Contour SGLD on a simple bimodal gaussian target. 
+++

# A one-dimensional Gaussian mixture with deep energy barriers

Let us first generate data points that follow a gaussian mixture distributions. 

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def gaussian_mixture_model(mu=-5.0, sigma=5.0, gamma=20.0):
    def sample_fn(rng_key, num_samples):
        key1, key2, key3 = jax.random.split(rng_key, 3)
        prob_mixture = jax.random.bernoulli(key1, p=0.5, shape=(num_samples, 1))
        mixture_1 = jax.random.normal(key2, shape=(num_samples, 1)) * sigma + mu
        mixture_2 = jax.random.normal(key3, shape=(num_samples, 1)) * sigma + gamma - mu
        return prob_mixture * mixture_1 + (1 - prob_mixture) * mixture_2

    def logprior_fn(position):
        return 0

    def loglikelihood_fn(position, x):
        mixture_1 = jax.scipy.stats.norm.logpdf(x, loc=position, scale=sigma)
        mixture_2 = jax.scipy.stats.norm.logpdf(x, loc=-position + gamma, scale=sigma)
        return jsp.special.logsumexp(jnp.array([mixture_1, mixture_2])) + jnp.log(0.5)

    return sample_fn, logprior_fn, loglikelihood_fn


sample_fn, logprior_fn, loglikelihood_fn = gaussian_mixture_model()
```

```{code-cell} ipython3
data_size = 1000

rng_key = jax.random.PRNGKey(0)
rng_key, sample_key = jax.random.split(rng_key)
X_data = sample_fn(sample_key, data_size)


import matplotlib.pylab as plt

fig = plt.figure(figsize=(5, 3))
ax = plt.subplot(111)
ax.hist(X_data.squeeze(), 100)
ax.set_xlabel("X")
ax.set_xlim(left=-15, right=35)

ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.title("Data Distribution")
```

From the figure above, we can see the **energy barrier** between the two modes is quite **high**, which leads to a hard problem. 

## Stochastic gradient Langevin dynamics

```{code-cell} ipython3
from fastprogress import progress_bar

import blackjax
import blackjax.sgmcmc.gradients as gradients

# Specify hyperparameters for SGLD
total_iter = 50_000
thinning_factor = 10

batch_size = 100
lr = 1e-3
temperature = 50.0

init_position = 10.0


# Build the SGLD sampler
grad_fn = gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)
sgld = blackjax.sgld(grad_fn)


# Initialize and take one step using the vanilla SGLD algorithm
position = init_position
sgld_sample_list = jnp.array([])

pb = progress_bar(range(total_iter))
for iter_ in pb:
    rng_key, batch_key, sample_key = jax.random.split(rng_key, 3)
    data_batch = jax.random.shuffle(batch_key, X_data)[:batch_size, :]
    position = jax.jit(sgld)(sample_key, position, data_batch, lr, temperature)
    if iter_ % thinning_factor == 0:
        sgld_sample_list = jnp.append(sgld_sample_list, position)
        pb.comment = f"| position: {position: .2f}"
```

```{code-cell} ipython3
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

fig = plt.figure(figsize=(12, 2))

G = gridspec.GridSpec(1, 3)

# Trajectory
ax = plt.subplot(G[0, :2])
ax.plot(sgld_sample_list, label="SGLD")
ax.set_xlabel(f"Iterations (x{thinning_factor})")
ax.set_ylabel("X")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# Histogram
ax = plt.subplot(G[0, 2])
ax.hist(sgld_sample_list, 100)
ax.set_xlabel("X")
ax.set_xlim(left=-15, right=35)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.suptitle("Stochastic gradient Langevin dynamics (SGLD)")
```

There are two modes, but the SGLD algorithm only recovers one of them.

## Contour stochastic gradient Langevin dynamics

specify hyperparameters (zeta and sz are the only two hyperparameters to tune)

```{code-cell} ipython3
zeta = 2
sz = 10
temperature = 50
```

Set basic hyperparameters

```{code-cell} ipython3
lr = 1e-3
init_position = 10.0
```

The following parameters partition the energy space and no tuning is needed.

```{code-cell} ipython3
num_partitions = 100000
energy_gap = 0.25
```

Initialize CSGLD

```{code-cell} ipython3
logdensity_fn = gradients.logdensity_estimator(logprior_fn, loglikelihood_fn, data_size)
csgld = blackjax.csgld(
    logdensity_fn,
    zeta=zeta,  # can be specified at each step in lower-level interface
    temperature=temperature,  # can be specified at each step
    num_partitions=num_partitions,  # cannot be specified at each step
    energy_gap=energy_gap,  # cannot be specified at each step
    min_energy=0,
)
```

Simulate via the CSGLD algorithm


```{code-cell} ipython3
state = csgld.init(init_position)

csgld_sample_list, csgld_energy_idx_list = jnp.array([]), jnp.array([])

pb = progress_bar(range(total_iter))
for iter_ in pb:
    rng_key, subkey = jax.random.split(rng_key)
    stepsize_SA = min(1e-2, (iter_ + 100) ** (-0.8)) * sz

    data_batch = jax.random.shuffle(rng_key, X_data)[:batch_size, :]
    state = jax.jit(csgld.step)(subkey, state, data_batch, lr, stepsize_SA)

    if iter_ % thinning_factor == 0:
        csgld_sample_list = jnp.append(csgld_sample_list, state.position)
        csgld_energy_idx_list = jnp.append(csgld_energy_idx_list, state.energy_idx)
        pb.comment = (
            f"| position {state.position: .2f}"
        )
```

Contour SGLD takes inspiration from the Wang-Landau algorithm to learn the density of states of the model at each energy level, and uses this information to flatten the target density to be able to explore it more easily.

As a result, the samples returned by contour SGLD are not from the target density directly, and we need to resample them using the density of state as importance weights to get samples from the target distribution.

```{code-cell} ipython3
important_idx = jnp.where(state.energy_pdf > jnp.quantile(state.energy_pdf, 0.95))[0]
scaled_energy_pdf = (
    state.energy_pdf[important_idx] ** zeta
    / (state.energy_pdf[important_idx] ** zeta).max()
)

csgld_re_sample_list = jnp.array([])
for _ in range(5):
    rng_key, subkey = jax.random.split(rng_key)
    for my_idx in important_idx:
        if jax.random.bernoulli(rng_key, p=scaled_energy_pdf[my_idx], shape=None) == 1:
            samples_in_my_idx = csgld_sample_list[csgld_energy_idx_list == my_idx]
            csgld_re_sample_list = jnp.concatenate(
                (csgld_re_sample_list, samples_in_my_idx)
            )
```

```{code-cell} ipython3
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt

fig = plt.figure(figsize=(12, 2))

G = gridspec.GridSpec(1, 3)

# Trajectory
ax = plt.subplot(G[0, :2])
ax.plot(csgld_sample_list, label="SGLD")
ax.set_xlabel(f"Iterations (x{thinning_factor})")
ax.set_ylabel("X")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# Histogram before resampling
ax = plt.subplot(G[0, 2])
ax.hist(csgld_sample_list, 100, label="before resampling", color='bisque')
ax.hist(csgld_re_sample_list, 100, label="after resampling")

ax.set_xlabel("X")
ax.set_ylabel("Frequency")
ax.set_xlim(left=-15, right=35)

ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend()
plt.suptitle("Contour SGLD")
```

## Why does Contour SGLD work?

The energy density is crucial for us to build a flat density, so let's take a look at the estimation returned by the algorithm. For illustration purposes, we smooth out fluctations and focus on the energy range from 3700 to 100000, which covers the major part of sample space.

```{code-cell} ipython3
smooth_energy_pdf = jnp.convolve(
    state.energy_pdf, jsp.stats.norm.pdf(jnp.arange(-100, 101), scale=10), mode="same"
)
interested_idx = jax.lax.floor((jnp.arange(3700, 10000)) / energy_gap).astype(
    "int32"
)  # min 3681

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(
    jnp.arange(num_partitions)[interested_idx] * energy_gap,
    smooth_energy_pdf[interested_idx],
)

ax.set_xlabel("Energy")
ax.set_ylabel("Energy Density")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()
```

From the figure above, we see that low-energy regions usually lead to much higher probability mass. Moreover, the slope is negative with a higher scale in low energy regions. In view of Eq.(8) in [the paper]( https://proceedings.neurips.cc/paper/2020/file/b5b8c484824d8a06f4f3d570bc420313-Paper.pdf), we can expect a **negative learning rate** to help the particle escape the local trap. Eventually, a particle is able to bounce out of the deep local traps freely instead of being absorbed into it.


# A two-dimensional mixture with 25 modes

```{code-cell} ipython3
import itertools

lmbda = 1/25
positions = [-4, -2, 0, 2, 4]
mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
sigma = 0.03 * jnp.eye(2)

def logprob_fn(x, *_):
    return lmbda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, mu, sigma))

def sample_fn(rng_key):
    choose_key, sample_key = jax.random.split(rng_key)
    samples = jax.random.multivariate_normal(sample_key, mu, sigma)
    return jax.random.choice(choose_key, samples)
```

```{code-cell} ipython3
from scipy.stats import gaussian_kde

rng_key = jax.random.PRNGKey(0)
samples = jax.vmap(sample_fn)(jax.random.split(rng_key, 10_000))

xmin, ymin = -5, -5
xmax, ymax = 5, 5

nbins = 300j
x, y = samples[:, 0], samples[:, 1]
xx, yy = jnp.mgrid[xmin:xmax:nbins, ymin:ymax:nbins]
positions = jnp.vstack([xx.ravel(), yy.ravel()])
values = jnp.vstack([x, y])
kernel = gaussian_kde(values)
f = jnp.reshape(kernel(positions).T, xx.shape)

fig, ax = plt.subplots()
cfset = ax.contourf(xx, yy, f, cmap='Blues')
ax.imshow(jnp.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')

plt.rcParams['axes.titlepad'] = 15.
plt.title("Samples from a mixture of 25 normal distributions")
```

## SGLD baseline
```{code-cell} ipython3
# 250k iterations
num_training_steps = 250000
schedule_fn = lambda k: 0.05 * k ** (-0.55)
schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

grad_fn = lambda x, _: jax.grad(logprob_fn)(x)
sgld = blackjax.sgld(grad_fn)

rng_key = jax.random.PRNGKey(3)
init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

position = init_position
sgld_samples = []
for i in progress_bar(range(num_training_steps)):
    _, rng_key = jax.random.split(rng_key)
    position = jax.jit(sgld)(rng_key, position, 0, schedule[i])
    sgld_samples.append(position)
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(111)
x = [sample[0] for sample in sgld_samples[::10]]
y = [sample[1] for sample in sgld_samples[::10]]

ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])

plt.axis('off')
```

## Contour SGLD

```{code-cell} ipython3
from typing import NamedTuple

import blackjax
from blackjax.types import Array, PRNGKey, PyTree

import seaborn as sns
```

hyperparameters in general

```{code-cell} ipython3
lr = 0.01
temperature = 1
```

hyperparameters for csgld

```{code-cell} ipython3
zeta = 0.06
sz = 10
```

The following parameters partition the energy space and no tuning is needed. 

```{code-cell} ipython3
cur_seed = 0
num_partitions = 10000
energy_gap = 0.001
```

Define functions 

```{code-cell} ipython3
lmbda = 1/25
positions = [-4, -2, 0, 2, 4]
mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
sigma = 0.03 * jnp.eye(2)

def logprob_fn(x, *_):
    return lmbda * jsp.special.logsumexp(
        jax.scipy.stats.multivariate_normal.logpdf(x, mu, sigma)
    )

def sample_fn(rng_key):
    choose_key, sample_key = jax.random.split(rng_key)
    samples = jax.random.multivariate_normal(sample_key, mu, sigma)
    return jax.random.choice(choose_key, samples)
```

Initialize CSGLD sampler

```{code-cell} ipython3
# 250k iterations
num_training_steps = 250000

grad_fn = lambda x, _: jax.grad(logprob_fn)(x)

rng_key = jax.random.PRNGKey(cur_seed)
init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

position = init_position

min_energy = -2 # min energy -1.2 max energy around 0.07

thinning_factor = 10

csgld = blackjax.csgld(
    logprob_fn,
    zeta=zeta,  # can be specified at each step in lower-level interface
    temperature=temperature,  # can be specified at each step
    num_partitions=num_partitions,  # cannot be specified at each step
    energy_gap=energy_gap,  # cannot be specified at each step
    min_energy=min_energy,
)
```

Simulate via the CSGLD algorithm

```{code-cell} ipython3 
state = csgld.init(init_position)

csgld_samples = [] 

for iter_ in progress_bar(range(num_training_steps)):
    rng_key, subkey = jax.random.split(rng_key)
    stepsize_SA = min(1e-2, (iter_+100)**(-0.8)) * sz
    state = jax.jit(csgld.step)(subkey, state, 0, lr, stepsize_SA)
    csgld_samples.append(state.position)

csgld_samples = jnp.array(csgld_samples)[1000: ][:: thinning_factor] # remove warm-up samples + thinning
```

Trajectory with Contour SGLD

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(111)
x = [sample[0] for sample in csgld_samples]
y = [sample[1] for sample in csgld_samples]

ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])

plt.axis('off')
plt.title("Trajectory with Contour SGLD")
```

Re-sampling via importance sampling

```{code-cell} ipython3
csgld_energy_idx_list = []
# get index of each particle for importance sampling
for position in progress_bar(csgld_samples):
    energy_value = logprob_fn(position)
    idx = jax.lax.min(jax.lax.max(jax.lax.floor((energy_value - min_energy) / energy_gap + 1).astype("int32"), 1,), num_partitions - 1, )
    csgld_energy_idx_list.append(idx)

csgld_energy_idx_list = jnp.array(csgld_energy_idx_list)


# pick important partitions for re-sampling
important_idx = jnp.where(state.energy_pdf > jnp.quantile(state.energy_pdf, 0.95))[0]
scaled_energy_pdf = state.energy_pdf[important_idx]**zeta / (state.energy_pdf[important_idx]**zeta).max()

csgld_re_sample_list = jnp.empty([0, 2])
for _ in range(5):
    rng_key, subkey = jax.random.split(rng_key)
    for my_idx in important_idx:
        if jax.random.bernoulli(rng_key, p=scaled_energy_pdf[my_idx], shape=None) == 1:
            samples_in_my_idx = csgld_samples[csgld_energy_idx_list == my_idx]
            csgld_re_sample_list = jnp.concatenate(
                (csgld_re_sample_list, samples_in_my_idx), axis=0
            )
```

Demo

```{code-cell} ipython3
import numpy as np

x = [sample[0] for sample in csgld_re_sample_list]
y = [sample[1] for sample in csgld_re_sample_list]
xx, yy = np.mgrid[xmin:xmax:nbins, ymin:ymax:nbins]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig, ax = plt.subplots()
cfset = ax.contourf(xx, yy, f, cmap='Blues')
ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')

plt.title("Contour SGLD Samples from a mixture of 25 normal distributions")
```