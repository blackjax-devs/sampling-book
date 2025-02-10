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

MCMC algorithms can be run in parallel (ensemble), in the sense of running multiple chains at once. During the phase where all chain have converged to the typical set, this parallelism improves wallclock time by a factor of the number of chains. This is because each chain draws samples just as well as any other, so we get more samples in the same time.

Reaching the typical set, on the other hand, is not as easily parallelizable, and for ensemble methods, this is the bottlenech. EMAUS is one algorithm, based on (microcanonical)[https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html] dynamics, designed to target this problem.

The idea is to run a batch (or ensemble) of chains of microcanonical dynamics without MH adjustment first, and based on convergence diagnostics, to switch all the chains to be adjusted. Without adjustment, microcanonical dynamics converge fast to the target, and with adjustment, the chains are guaranteed to be asymptotically unbiased.

This code is designed to be run on GPU, and even across multiple nodes.

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

 

def run_emaus(
          chains= 4096, 
          alpha = 1.9, C= 0.1,
          early_stop=1, 
          r_end= 1e-2, # switch parameters
          diagonal_preconditioning= 1, 
          steps_per_sample= 15, 
          acc_prob= None # adjusted parameters
          ):
    
    key = jax.random.split(jax.random.key(42), 100)[2]
          
    info, grads_per_step, _acc_prob, final_state = emaus(
    
        logdensity_fn=logdensity_fn, 
        sample_init=sample_init, 
        ndims=2, 
        num_steps1=100, 
        num_steps2=300, 
        num_chains=chains, 
        mesh=mesh, 
        rng_key=key, 
        alpha= alpha, 
        C= C, 
        early_stop= early_stop, 
        r_end= r_end,
        diagonal_preconditioning= diagonal_preconditioning, 
        integrator_coefficients= None, 
        steps_per_sample= steps_per_sample, 
        acc_prob= acc_prob,
        ensemble_observables= lambda x: x
        ) 
    
    return final_state.position

samples = run_emaus()
```

The above code runs EMAUS with 4096 chains, on a banana shaped density function, and returns only the final step of each chain. These can be plotted:

```{code-cell} ipython3

import seaborn as sns
sns.scatterplot(x= samples[:, 0], y= samples[:, 1], alpha= 0.1)
```
