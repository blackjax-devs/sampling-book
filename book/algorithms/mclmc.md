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

An example is given below.

```{code-cell} ipython3
import blackjax
import jax
import jax.numpy as jnp
```

```{code-cell} ipython3
def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.mclmc.init(
        x_initial=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.noneuclidean_mclachlan,
        transform=lambda x: x,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
    )

    keys = jax.random.split(run_key, num_steps)

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
        transform=transform,
    )

    _, blackjax_mclmc_result = jax.lax.scan(
        f=lambda state, k: sampling_alg.step(
            rng_key=k,
            state=state,
        ),
        xs=keys,
        init=blackjax_state_after_tuning,
    )

    return blackjax_mclmc_result.transformed_position
```

```{code-cell} ipython3
samples = run_mclmc(
    logdensity_fn= lambda x : -0.5 * jnp.sum(jnp.square(x)), 
    num_steps=1000, initial_position=jnp.ones((1000,)), 
    key=jax.random.PRNGKey(0),
    transform=lambda x: x[:2],)
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
