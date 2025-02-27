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

# Ensemble Microcanonical Adjusted-Unadjusted Sampler (EMAUS)


MCMC algorithms can converge in significantly lower wallclock time if instead of running one long chain that collects many samples, we in parallel run multiple short chains, each only collecting one effective sample. The bottleneck of this approach is the burn-in, because it determines when the chains produce the first effective sample. EMAUS is one such parallel algorithm which is particularly fast. It is based on (microcanonical)[https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html]
dynamics which excels in fast burn-in. Another important speed-up over the other methods is that chains are initially run without MH adjustment, which we find to be faster during the burn-in. Later, based on convergence diagnostics, Metropolis Adjustment is switched on which speeds up fine convergence and guarantees asymptotically unbiased samples.

This code is designed to be run on CPU or GPU, and even across multiple nodes.

```{code-cell} ipython3

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from blackjax.adaptation.ensemble_mclmc import emaus


mesh = jax.sharding.Mesh(jax.devices(), 'chains')

sample_init = lambda key: jax.random.normal(key, shape=(2,)) * jnp.array([10.0, 5.0]) * 2

def logdensity_fn(x):
    mu2 = 0.03 * (x[0] ** 2 - 100)
    return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

info, grads_per_step, _acc_prob, final_state = emaus(
    
        logdensity_fn=logdensity_fn, 
        sample_init=sample_init, 
        transform=lambda x:x,
        ndims=2, 
        num_steps1=100, 
        num_steps2=300, 
        num_chains=512, 
        mesh=mesh, 
        rng_key=jax.random.key(42), 
        early_stop=True, # allow the unadjusted phase to end early, based on a cross-chain convergence criterion 
        diagonal_preconditioning=True, 
        integrator_coefficients= None, 
        steps_per_sample=15, # number of steps in proposals in adjusted phase
        ensemble_observables= lambda x: x
        ) 
    
samples = final_state.position
```

The above code runs EMAUS with 512 chains, on a banana shaped density function, and returns only the final state of each chain. These can be plotted:

```{code-cell} ipython3
import seaborn as sns
sns.scatterplot(x= samples[:, 0], y= samples[:, 1], alpha= 0.1)
```

```{code-cell} ipython3

```
