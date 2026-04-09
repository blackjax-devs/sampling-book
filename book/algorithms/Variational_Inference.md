---
jupytext:
  cell_metadata_filter: -all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  formats: md:myst
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

# Variational Inference

## 1. The idea behind variational inference

This chapter is an introduction to variational inference. Additional information about variational inference can be found here: https://arxiv.org/abs/1601.00670

A model represents an approximation of a real-world scenario at a certain level, incorporating some assumptions. In statistics, it is a mathematical framework that describes a process using both data and assumptions. These models include parameters that are unknown and must be estimated from the available data to reveal hidden factors or predict outcomes. If a model does not align with the data, it can be rejected in favor of a better one. Statistical inference is the process of using observed data to draw conclusions about the underlying probability distribution. In conclusion, a probabilistic model $p(x)$
represents a probability distribution over a representative dataset $\{x_1, x_2, \dots, x_n\}$, and modeling $p(x)$ means fitting a distribution to this data. It is also important to note that probabilistic models can be conditional, such as $p(y \mid x)$. 

Since posterior distributions often become highly complex and high-dimensional, it makes analytical computations challenging and intractable. One practical solution to this problem is approximation. A function can be understood as a mapping that takes an input variable and returns an output value. Its derivative
indicates how the output value changes with small adjustments to the input. Extending this idea, we consider an operator that accepts an entire function as
input and returns a numerical value as output. Many problems can thus be expressed as optimization problems involving these operators, where the objective is to find the input function that minimizes or maximizes the operator’s output. Approximate solutions typically limit the class of functions explored during optimization, simplifying the computational process. One of these solutions is variational inference.

The objective of variational inference is to approximate the conditional distribution of latent variables given observed data. This is achieved by formulating inference as an optimization problem. Specifically, one posits a family of tractable distributions (e.g., Gaussians) over the latent variables, which have a set of parameter values within that family. The optimal member of this family is identified by minimizing the Kullback-Leibler (KL) divergence between the candidate distribution and the true posterior distribution of interest. The resulting variational distribution then serves as a surrogate for the exact posterior, enabling approximate inference that is computationally efficient and scalable.

By taking a closer look at the following equation:

$$
p(z \mid x) = \frac{p(x, z)}{p(x)}
$$

Where

- $p(z \mid x)$ is unknown
- $p(x, z)$ is known
- $p(x)$ is intractable to compute

Despite the intractability of $p(x)$, it is still possible to approximate the unknown $p(z \mid x)$ by introducing a function $q(z)$ which aims to approximate $p(z \mid x)$. Let KL-divergence, which is the measure of the difference between two probability distributions, be presented as:

$$
KL = - \sum q(z) * \log \frac{p(z \mid x)}{q(x)}
$$

+++

Let $L$ be the evidence lower bound on the log-likelihood of the observed data discussed in Appendix B, and be presented as:

$$
L = \sum q(z) * \log \frac{p(x, z)}{q(z)}
$$

so that

$$
KL + L = \ln p(x)
$$

By increasing the lower bound, we reduce the KL divergence. Maximizing the lower bound is often more practical, since the KL divergence involves the joint distribution, whereas the lower bound only requires the joint probability in its numerator. Because we want the KL divergence to be as small as possible, the goal is to make the lower bound as large as possible. Therefore, the key idea is to find a distribution $q(z)$ that maximizes the lower bound. This approach forms the foundation of variational inference. By selecting a tractable form for $q(z)$, the inference problem becomes computationally feasible. (https://studenttheses.uu.nl/handle/20.500.12932/50019?show=full)









## 2. Different types of objectives
In most cases, variational inference is presented with the Kullback-Leibler (KL) divergence as the objective. However, the literature also studies alternative objectives. One important example is the variational Rényi (VR) bound, which extends traditional variational inference to Rényi's alpha-divergences.

For two distributions \(p\) and \(q\), Rényi's alpha-divergence is defined as

$$
D_\alpha[p \| q]
=
\frac{1}{\alpha - 1}
\log \int p(z)^\alpha q(z)^{1-\alpha}\,dz,
\qquad \alpha > 0,\ \alpha \neq 1.
$$

As alpha = 1, this expression recovers the Kullback-Leibler divergence. In variational inference, this leads to the variational Rényi bound

$$
\mathcal{L}_\alpha(q; x)
=
\frac{1}{1-\alpha}
\log
\mathbb{E}_{q(z)}
\left[
\left(
\frac{p(x,z)}{q(z)}
\right)^{1-\alpha}
\right].
$$

When alpha = 1, the variational Rényi bound reduces to the standard evidence lower bound (ELBO). Different values of alpha change the behavior of the approximation, allowing a trade-off between mode-seeking and mass-covering behavior. (https://arxiv.org/abs/1602.02311)

Tail-adaptive 𝑓-divergence is a variational inference objective designed to keep the mass-covering advantages of α-divergences while avoiding the instability that can arise when importance weights have heavy tails. The key idea is to adapt the divergence to the tail behavior of the density ratio during training, which makes optimization more stable and can improve performance on complex, multimodal problems. (https://arxiv.org/abs/1810.11943)

A further generalization is given by the scale-invariant alpha-beta (sAB) divergence. This is a two-parameter family of objectives that includes several well-known divergence-based approaches as special cases, including KL, Rényi, and gamma-type objectives. In variational inference, this objective is optimized directly as a divergence between the variational approximation and the posterior distribution. Its additional flexibility allows one to control both the mass-covering versus mode-seeking behavior of the approximation and its robustness to outliers. (https://arxiv.org/abs/1805.01045)

Robustness to outliers is something that was considered by researchers regarding variational inference. As a result, a few articles replace KL divergence with wth a robust divergences such as $\beta$ and $\gamma$. (https://arxiv.org/abs/1710.06595). 

χ-divergence variational inference replaces the usual KL-based objective with a divergence that encourages more overdispersed approximations and avoids assigning too little mass to regions where the true posterior is nonzero. In this framework, inference is performed by minimizing the χ upper bound (CUBO), which provides an upper bound on the model evidence. (https://arxiv.org/abs/1611.00328)









## 3. The use of variational inference in practice 

As an example, we set up a variational inference experiment to approximate a posterior distribution over a 3‑dim variable. The target posterior combines a uniform prior on [−10, 10] with a Gaussian mixture likelihood (3 components) fitted to synthetically generated data. We also compare two variational families: mean‑field (MFVI) and full‑rank (FRVI), and two divergences: KL and Rényi-α. After sampling 10,000 draws from each approximation and ground-truth generated data, we evaluate the quality by computing means, variances, KL divergence metric to the ground truth posterior and visualise all distributions with a corner plot. The goal is to see which combination of family and divergence yields the most accurate posterior approximation.

```{code-cell} python3
import blackjax
from blackjax.vi.fullrank_vi import as_top_level_api as frvi_top_level_api
from blackjax.vi.meanfield_vi import as_top_level_api as mfvi_top_level_api
from blackjax.vi._gaussian_vi import KL, RenyiAlpha

from typing import Tuple
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
import optax
from jax import random
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

```











## 4. Likelihood

The `GaussianMixtureLikelihood` class computes the log-likelihood of a point under a pre‑fitted Gaussian mixture model. It pre‑computes Cholesky factors, log‑weights, and log‑normalization constants for efficiency. `loglike_single` method returns the log‑likelihood, which is added to a uniform log‑prior to form the target posterior for variational inference.


```{code-cell} python3
Array = jax.Array

class GaussianMixtureLikelihood:
    """Defines Gaussian Mixture Likelihood"""
    def __init__(self, gmm):
        means = jnp.asarray(gmm.means_)
        covs = jnp.asarray(gmm.covariances_)
        weights = jnp.asarray(gmm.weights_)

        self.means = means
        self.chols = jax.vmap(jnp.linalg.cholesky)(covs)

        log_dets = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(self.chols, axis1=-2, axis2=-1)),
            axis=-1,
        )

        d = means.shape[1]
        self.log_w = jnp.log(weights / jnp.sum(weights))
        self.log_norms = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_dets)

    def loglike_single(self, x: Array) -> Tuple[Array, Array]:
        diffs = x - self.means

        def quad_form(diff, L):
            y = solve_triangular(L, diff, lower=True)
            return jnp.sum(y * y)

        quad = jax.vmap(quad_form)(diffs, self.chols)
        ll = logsumexp(self.log_w + self.log_norms - 0.5 * quad)
        return ll, jnp.zeros((0,), dtype=ll.dtype)

```








## 5. Experiment setup

Next we set up the synthetic ground truth: 10000 3‑dim points are generated from three Gaussian blobs, then a full‑covariance Gaussian mixture model is fitted to them to serve as the true posterior. The target log‑density for variational inference is defined. `run_vi_experiment` function then runs variational inference using mean‑field and full‑rank Gaussian families, optimizes the chosen divergence (KL and Rényi) with Adam over 1000 steps, and returns posterior samples.


```{code-cell} python3
# define parameters
seed = 847
n_components = 3
dim = 3
n = 10_000

# generate data with scikit-learn
target_data, blob_labels = make_blobs(
    n_samples=n,
    centers=n_components,
    n_features=dim,
    cluster_std=[1.0, 1.4, 0.8] if n_components == 3 else [1.0, 1.4],
    center_box=(-3.0, 3.0),
    random_state=seed,
)

print(np.unique(blob_labels, return_counts=True))

# fit a Gaussian mixture model to the synthetic data
gmm = GaussianMixture(
    n_components=n_components,
    covariance_type="full",
    random_state=seed,
)
gmm.fit(target_data)

# samples from fitted GMM 
true_samples_raw, _ = gmm.sample(10_000)

# define likelihood
likelihood = GaussianMixtureLikelihood(gmm)

# define bounds
low = jnp.array([-10.0, -10.0, -10.0])
high = jnp.array([10.0, 10.0, 10.0])

# define log-density: uniform prior and GMM likelihood
def logdensity_fn(x):
    lp = jnp.where(jnp.all((x >= low) & (x <= high)), 0.0, -jnp.inf)
    ll, _ = likelihood.loglike_single(x)
    return lp + ll


def run_vi_experiment(
    vi_family,
    objective,
    seed=0,
    num_steps=1000,
    num_mc_samples=100,
    num_draws=10_000,
    learning_rate=1e-2,
):
    optimizer = optax.adam(learning_rate)

    # STL is used for KL only
    if isinstance(objective, RenyiAlpha) and objective.alpha != 1.0:
        stl_estimator = False
    else:
        stl_estimator = True

    if vi_family == "mfvi":
        vi_algo = mfvi_top_level_api(
            logdensity_fn,
            optimizer,
            num_samples=num_mc_samples,
            objective=objective,
            stl_estimator=stl_estimator,
        )
    elif vi_family == "frvi":
        vi_algo = frvi_top_level_api(
            logdensity_fn,
            optimizer,
            num_samples=num_mc_samples,
            objective=objective,
            stl_estimator=stl_estimator,
        )
    else:
        raise ValueError("Unknown vi_family")
    # define optimizer
    key = random.PRNGKey(seed)
    state = vi_algo.init(jnp.zeros(dim))

    losses = []
    for _ in range(num_steps):
        key, subkey = random.split(key)
        state, info = vi_algo.step(subkey, state)
        losses.append(info)

    key, subkey = random.split(key)
    samples = np.array(vi_algo.sample(subkey, state, num_draws))

    return samples, state, losses

```








## 6. Sampling

This code block runs four variational inference experiments: mean‑field and full‑rank VI each with KL divergence, and Rényi-α divergence. Each call to run_vi_experiment optimizes the variational approximation over 1000 steps, returning posterior samples. The resulting samples are later compared against the true posterior drawn from the fitted GMM.

```{code-cell} python3
# define objectives
kl = KL()
renyi = RenyiAlpha(alpha=0.5)

# sample
mfvi_kl_samples, mfvi_kl_state, mfvi_kl_losses = run_vi_experiment("mfvi", kl, seed=0)
frvi_kl_samples, frvi_kl_state, frvi_kl_losses = run_vi_experiment("frvi", kl, seed=0)
mfvi_renyi_samples, mfvi_renyi_state, mfvi_renyi_losses = run_vi_experiment("mfvi", renyi, seed=0)
frvi_renyi_samples, frvi_renyi_state, frvi_renyi_losses = run_vi_experiment("frvi", renyi, seed=0)

```








## 7. Define diagnostics

Next code defines a `compute_metrics` function that calculates the sample mean and variance of both the variational posterior samples and the true posterior samples, then computes the KL divergence between generated true posterior and posterior samples from variational inference. The function is called on all four variational approximations (MFVI+KL, FRVI+KL, MFVI+Rényi, FRVI+Rényi) to produce comparisons of their accuracy relative to the ground truth.

```{code-cell} python3

# function for computing means, variances of true samples and VI samples
def compute_metrics(vi_samples, true_samples):
    """Compute statistics and KL divergence."""
    vi_samples = np.array(vi_samples)
    true_samples = np.array(true_samples)
    
    vi_mean = vi_samples.mean(axis=0)
    vi_var = vi_samples.var(axis=0)
    true_mean = true_samples.mean(axis=0)
    true_var = true_samples.var(axis=0)
    
    # KL divergence metric
    def gau_kl(pm, pv, qm, qv):
        axis = 1 if len(qm.shape) == 2 else 0
        dpv = pv.prod()
        dqv = qv.prod(axis)
        iqv = 1. / qv
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)
            + (iqv * pv).sum(axis)
            + (diff * iqv * diff).sum(axis)
            - len(pm)
        ))
    
    kl_div = gau_kl(vi_mean, vi_var, true_mean, true_var)
    
    return {'vi_mean': vi_mean, 'vi_var': vi_var,
        'true_mean': true_mean, 'true_var': true_var,
        'kl_divergence': kl_div}


```


```{code-cell} python3
# compute specified metrics
mfvi_kl_metrics = compute_metrics(mfvi_kl_samples, true_samples_raw)
frvi_kl_metrics = compute_metrics(frvi_kl_samples, true_samples_raw)
mfvi_renyi_metrics = compute_metrics(mfvi_renyi_samples, true_samples_raw)
frvi_renyi_metrics = compute_metrics(frvi_renyi_samples, true_samples_raw)

```








## 8. Comparing means

```{code-cell} python3
true_mean = np.asarray(true_samples_raw).mean(axis=0)
print("\nTABLE 1: MEANS COMPARISON")
print(pd.DataFrame({
        "Variable": [f"x{i+1}" for i in range(3)], "True Mean": true_mean,
        "MFVI + KL": mfvi_kl_metrics["vi_mean"], "FRVI + KL": frvi_kl_metrics["vi_mean"],
        "MFVI + Renyi": mfvi_renyi_metrics["vi_mean"], "FRVI + Renyi": frvi_renyi_metrics["vi_mean"],
    }).round(6).to_string(index=False))

```







## 9. Comparing variances

```{code-cell} python3
true_var = np.asarray(true_samples_raw).var(axis=0)
print("\nTABLE 2: VARIANCES COMPARISON")
print(
    pd.DataFrame({
        "Variable": [f"x{i+1}" for i in range(3)], "True Variance": true_var,
        "MFVI + KL": mfvi_kl_metrics["vi_var"], "FRVI + KL": frvi_kl_metrics["vi_var"],
        "MFVI + Renyi": mfvi_renyi_metrics["vi_var"], "FRVI + Renyi": frvi_renyi_metrics["vi_var"],
    }).round(6).to_string(index=False))

```





## 10. Comparing KL DIVERGENCE

```{code-cell} python3

print("\nTABLE 3: KL DIVERGENCE")
print(
    pd.DataFrame({
        "Method": [ "MFVI + KL", "FRVI + KL",
            "MFVI + Renyi", "FRVI + Renyi",],
        "KL Divergence": [mfvi_kl_metrics["kl_divergence"], frvi_kl_metrics["kl_divergence"],
                         mfvi_renyi_metrics["kl_divergence"], frvi_renyi_metrics["kl_divergence"],],
    }).round(6).to_string(index=False))


```





## 11. Visual diagnostics

The plot enables visual comparison of how well each variational family (mean‑field vs. full‑rank) and divergence (KL vs. Rényi) captures the true distributions.



```{code-cell} python3

# convert to arrays 
true_samples_raw = np.asarray(true_samples_raw)

# all VI samples for comparison
frvi_kl_samples_arr = np.asarray(frvi_kl_samples)
mfvi_kl_samples_arr = np.asarray(mfvi_kl_samples)
frvi_renyi_samples_arr = np.asarray(frvi_renyi_samples)
mfvi_renyi_samples_arr = np.asarray(mfvi_renyi_samples)

d = true_samples_raw.shape[1]
cols = [f"x{i+1}" for i in range(d)]
# dataframe with all samples
df_true = pd.DataFrame(true_samples_raw, columns=cols)
df_true["Source"] = "True Gaussian"

# all four VI methods
df_frvi_kl = pd.DataFrame(frvi_kl_samples_arr, columns=cols)
df_frvi_kl["Source"] = "FRVI + KL"
df_mfvi_kl = pd.DataFrame(mfvi_kl_samples_arr, columns=cols)
df_mfvi_kl["Source"] = "MFVI + KL"
df_frvi_renyi = pd.DataFrame(frvi_renyi_samples_arr, columns=cols)
df_frvi_renyi["Source"] = "FRVI + Renyi"
df_mfvi_renyi = pd.DataFrame(mfvi_renyi_samples_arr, columns=cols)
df_mfvi_renyi["Source"] = "MFVI + Renyi"
# all dataframes combined
df_all = pd.concat([df_true, df_frvi_kl, df_mfvi_kl, df_frvi_renyi, df_mfvi_renyi], ignore_index=True)

# Create corner-like comparison plot
g = sns.pairplot( 
    df_all, vars=cols, hue="Source",
    corner=True, diag_kind="kde", kind="kde",                        
    plot_kws={"fill": False, "levels": 5}, 
    diag_kws={"fill": False})

g.fig.suptitle("Distribution Comparison: True vs All VI Methods", fontsize=8, fontweight="bold", y=1.02)
plt.show()


```

