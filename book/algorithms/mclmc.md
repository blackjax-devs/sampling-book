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

This is an algorithm based on https://arxiv.org/abs/2212.08549. The original derivation comes from thinking about the microcanonical ensemble (a concept from statistical mechanics), but the upshot is that we integrate the following SDE:

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

where $u$ is an auxilliary variable, $S(x)$ is the PDF of the distribution from which we are sampling and the last term describes spherically symmetric noise. After $u$ is marginalized out, this converges to the target distribution $S(x)$.

## How to run MCLMC in BlackJax

It is very important to use the tuning algorithm provided, which controls the step size of the integrator and also $L$, a parameter related to $\eta$ above.

An example is given below, of a 1000 dim Gaussian (of which 2 dimensions are plotted).

```{code-cell} ipython3
import blackjax
import jax
import jax.numpy as jnp
```

```{code-cell} ipython3
def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        x_initial=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.noneuclidean_mclachlan,
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
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    # run the sampler
    _, samples, _ =  blackjax.util.run_inference_algorithm(
            rng_key=run_key,
            initial_state_or_position=blackjax_state_after_tuning,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=transform,
            progress_bar=True,
        )

    return samples
```

```{code-cell} ipython3
# run the algorithm on a high dimensional gaussian, and show two of the dimensions

samples = run_mclmc(
    logdensity_fn= lambda x : -0.5 * jnp.sum(jnp.square(x)), 
    num_steps=1000, initial_position=jnp.ones((1000,)), 
    key=jax.random.PRNGKey(0),
    transform=lambda x: x.position[:2],)
samples.mean()
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.axis('equal')
plt.scatter(x=samples[:,0], y=samples[:,1])
plt.title('Scatter Plot of Samples')
```

```{code-cell} ipython3

```
