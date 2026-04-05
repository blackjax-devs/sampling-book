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

Despite the intractability of $p(x)$, it is still possible to approximate the unknown $p(z \mid x)$ by introducing a function $q(z)$ which aims to approximate $p(z \mid x)$. Let KL-divergence, which is the measure of the difference between two probability distributions and discussed in Appendix A, be presented as:

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

By increasing the lower bound, we reduce the KL divergence. Maximizing the lower bound is often more practical, since the KL divergence involves the joint distribution, whereas the lower bound only requires the joint probability in its numerator. Because we want the KL divergence to be as small as possible, the goal is to make the lower bound as large as possible. Therefore, the key idea is to find a distribution $q(z)$ that maximizes the lower bound. This approach forms the foundation of variational inference. By selecting a tractable form for $q(z)$, the inference problem becomes computationally feasible.

Let's know take a look at some examples of how to use variational inference. To make an experiment, let's start with a Gaussian mixture generator which will randomly generates Gaussian with the specified number of components per dimension. 

## 2. Random Gaussian generator

```{code-cell} python3


import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner


"""The following can be used to improve plotting a bit
   Credits: https://github.com/ThibeauWouters"""
params = {
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16
}

plt.rcParams.update(params)

# Improved corner kwargs -- pass them to corner.corner
default_corner_kwargs = dict(bins=50, 
                        smooth=0.5, 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)


class GaussianMixtureGenerator:
    """Utility class to generate samples from a mixture of Gaussians."""

    @staticmethod
    def generate_gaussian_mixture(n_dim: int,
                                  n_gaussians: int = None,
                                  n_samples: int = None,
                                  means: list = None,
                                  covariances: list = None,
                                  weights: list = None,
                                  width_mean: float = None,
                                  width_cov: float = None):
        """
        Generate samples from a mixture of Gaussians. 
        This function generates samples from a Gaussian mixture model with specified means, covariances, and weights.
        If means, covariances, or weights are not provided, they are generated randomly.
        
        Args:
            n_dim (int): The number of dimensions for the samples.
            n_gaussians (int, optional): The number of Gaussian components in the mixture. Defaults to 1.
            n_samples (int, optional): The number of samples to generate. Defaults to 10,000.
            means (list, optional): The mean vectors. If not specified, then they will be generated randomly. Defaults to None.
            covariances (list, optional): The square covariance matrix of size (n_dim x n_dim). If not specified, then they will be generated randomly. Defaults to None.
            weights (list, optional): Weights between the different Gaussians. If not specified, equal weights are used. Defaults to None.
            width_mean (float, optional): The width of the mean distribution. Defaults to 10.0.
            width_cov (float, optional): The width of the covariance distribution. Defaults to 1.0.
        """
        
        # If no mean vector is given, generate random means
        seed = np.random.randint(0, 1000)
        jax_key = jax.random.PRNGKey(seed)
        if means is None:
            means = []
            for _ in range(n_gaussians):
                # Split the key to ensure different means for each Gaussian
                jax_key, subkey = jax.random.split(jax_key)
                this_means = jax.random.uniform(subkey, (n_dim,), minval=-width_mean, maxval=width_mean)
                #print("this_means")
                print(this_means)
                
                means.append(this_means)
        #print(f"Means: {means}")
            
        # If no covariance matrix is given, generate identity matrices
        if covariances is None:
            covariances = []
            for _ in range(n_gaussians):
                jax_key, subkey = jax.random.split(jax_key)
                A = jax.random.uniform(subkey, (n_dim, n_dim), minval=-width_cov, maxval=width_cov)
                B = jnp.dot(A, A.transpose())
                covariances.append(B)
        #print(f"Covariances: {covariances}")
        
        # If no weights are given, use equal weights between the Gaussians
        if weights is None:
            weights = [1.0 / n_gaussians] * n_gaussians
        #print(f"Weights: {weights}")
            
        # Check if everythingq is consistent
        if len(means) != n_gaussians or len(covariances) != n_gaussians or len(weights) != n_gaussians:
            raise ValueError("Means, covariances, and weights must match the number of Gaussians.")
        
        # Generate samples
        samples = []
        for i in range(n_samples):
            # Choose a Gaussian component based on weights
            this_key = jax.random.PRNGKey(i)
            this_key, sample_key = jax.random.split(this_key)
            component = np.random.choice(n_gaussians, p=weights)
            mean = means[component]
            covariance = covariances[component]
            
            # Generate a sample from the chosen Gaussian
            sample = jax.random.multivariate_normal(sample_key, mean, covariance)
            samples.append(sample)
            
        samples = jnp.array(samples)
        return samples, means, covariances, weights

...
```

## 3. Likelihood function
The next step is to define multivariate likelihood 


```{code-cell} python3



from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

Array = jax.Array


class GaussianMixtureParams(eqx.Module):
    """
    Parameters for a Gaussian Mixture Model.
    
    Parameters:
    ------------
        means: Mean vectors for each component, shape (K, D) where K = number of components, 
               D = data dimension
        chols: Lower Cholesky factors of covariance matrices, shape (K, D, D). Each chols[k] 
               is lower-triangular
        log_w: log weights (log mixing coefficients) for each component, shape (K,)
        log_norms: Log normalization constants for each Gaussian component, shape (K,).
    
    Equal to -0.5 * (D * log(2pi) + log(det(cov_k)))

    """
    means: Array      # (K, D)
    chols: Array      # (K, D, D)
    log_w: Array      # (K,)
    log_norms: Array  # (K,)


def gmm_init_params(
    means: Array,      # (K, D)
    covs: Array,       # (K, D, D)
    weights: Array,    # (K,)
    *,
    logits: bool = False,
    eps: float = 1e-30,
) -> GaussianMixtureParams:
    """Initialize Gaussian Mixture Model parameters from raw inputs.
    
    Parameters:
    -----------
        means: Component means, shape (K, D)
        covs: Component covariance matrices (must be positive definite), shape (K, D, D)
        weights: Component weights (mixing coefficients), shape (K,)
        logits: If True, weights as logits and apply log_softmax.
                If False, weights as probabilities and normalize to sum to 1 before taking log.
        eps: constant to avoid log(0) when weights=False
    """
    means = jnp.asarray(means)
    covs = jnp.asarray(covs)
    weights = jnp.asarray(weights)

    K, D = means.shape

    chols = jax.vmap(jnp.linalg.cholesky)(covs)  # (K, D, D)

    log_dets = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(chols, axis1=-2, axis2=-1)),
        axis=-1,
    )  # (K,)

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=means.dtype)
    log_norms = -0.5 * (jnp.asarray(D, dtype=means.dtype) * jnp.log(two_pi) + log_dets)  # (K,)

    log_w = jax.lax.cond(
        jnp.asarray(logits),
        lambda w: jax.nn.log_softmax(w),
        lambda w: jnp.log((w / jnp.sum(w)) + jnp.asarray(eps, dtype=w.dtype)),
        weights,
    )

    return GaussianMixtureParams(means=means, chols=chols, log_w=log_w, log_norms=log_norms)


def gmm_log_prob_single(params: GaussianMixtureParams, x: Array) -> Array:
    """Compute log probability of a single data point.
    
    Parameters:
    -----------
        params: GaussianMixtureParams with component parameters
        x: Single data point, shape (D,)
    
    Returns:
    ---------
        log p(x | params), scalar value
        """    
    # x: (D,)
    diffs = x - params.means  # (K, D)

    def quad_form(diff: Array, L: Array) -> Array:
        y = solve_triangular(L, diff, lower=True)
        return jnp.sum(y * y)

    quad = jax.vmap(quad_form)(diffs, params.chols)     # (K,)
    log_comp = params.log_norms - 0.5 * quad            # (K,)
    return logsumexp(params.log_w + log_comp)           # scalar


def gmm_log_prob(params: GaussianMixtureParams, xs: Array) -> Array:
    """Compute log prob for a batch of data points."""
    # xs: (..., D) -> (...,)
    xs = jnp.asarray(xs)
    D = params.means.shape[1]
    flat = xs.reshape((-1, D))
    flat_lp = jax.vmap(lambda z: gmm_log_prob_single(params, z))(flat)
    return flat_lp.reshape(xs.shape[:-1])




class GaussianMixtureLikelihood(eqx.Module):
    params: GaussianMixtureParams  

    def __init__(self, means, covs, weights, *, logits: bool = False):
        self.params = gmm_init_params(means=means, covs=covs, weights=weights, logits=logits)

    def loglike_single(self, x_1d: Array) -> Tuple[Array, Array]:
        ll = gmm_log_prob_single(self.params, x_1d)
        return ll, jnp.zeros((0,), dtype=ll.dtype)  # blobs_dim=0


...
```





## 4. Sampling

Now we are ready to sample. The idea is simple:

  (i) Gaussian distributions will be generated randomly in specified parameter space. 
  (ii) Mean and variance from Gaussian distributions will be passed to likelihood as estimators
  (iii) prior here is belief about bounds of the parameter space

```{code-cell} python3



# diagnostic packages
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random
import matplotlib.pyplot as plt
import corner
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# algorithms
import blackjax
import blackjax
from blackjax.vi.fullrank_vi import as_top_level_api as frvi_top_level_api
from blackjax.vi.meanfield_vi import as_top_level_api as mfvi_top_level_api
from blackjax.vi._gaussian_vi import KL, RenyiAlpha




print("="*80)
print("PART 1: TRAINING VARIATIONAL INFERENCE MODELS")
print("="*80)

# Set seed
np.random.seed(847)

# Generate target distribution
print("\n1.1 Generating target Gaussian distribution...")
true_samples_raw, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
    n_dim=3, n_gaussians=1, weights=[1.0], n_samples=10_000, width_mean=4.0, width_cov=1.0
)
true_samples_raw = np.array(true_samples_raw)
print(f" Generated {true_samples_raw.shape[0]} samples from {len(means)} component Gaussian")

# Initialize likelihood
likelihood = GaussianMixtureLikelihood(
    means=jnp.asarray(means),
    covs=jnp.asarray(covariances),
    weights=jnp.asarray(weights),
    logits=False,
)

# Define prior bounds
prior_bounds = {
    "x0": [-10.0, 10.0], "x1": [-10.0, 10.0], "x2": [-10.0, 10.0],
}

low = jnp.array([
    prior_bounds["x0"][0], prior_bounds["x1"][0], prior_bounds["x2"][0],
])

high = jnp.array([
    prior_bounds["x0"][1], prior_bounds["x1"][1], prior_bounds["x2"][1],
])

def logdensity_fn(x):
    lp = jnp.where(jnp.all((x >= low) & (x <= high)), 0.0, -jnp.inf)
    ll, _ = likelihood.loglike_single(x)
    return lp + ll

# Choose objective type
objective_name = "kl"  # "kl" or "renyi" or "tail_adaptive"
alpha = 0.5
tail_beta = -1.0

if objective_name == "kl":
    objective = KL()
    stl_estimator = True
    objective_tag = "kl"
elif objective_name == "renyi":
    objective = RenyiAlpha(alpha=alpha)
    stl_estimator = (alpha == 1.0)
    objective_tag = f"renyi_alpha_{str(alpha).replace('.', 'p')}"
#elif objective_name == "tail_adaptive":
    #objective = TailAdaptive(beta=tail_beta)
    #stl_estimator = True
    #objective_tag = f"tail_adaptive_beta_{str(tail_beta).replace('.', 'p').replace('-', 'neg')}"

print(f"\n1.2 Objective function: {objective_tag}")

# Function to run VI experiment
def run_vi_experiment(vi_family, logdensity_fn, low, high, objective, stl_estimator):
    """Run a single VI experiment and return samples."""
    
    optimizer = optax.adam(0.01)
    
    if vi_family == "mfvi":
        vi_algo = mfvi_top_level_api(
            logdensity_fn,
            optimizer,
            num_samples=100,
            objective=objective,
            stl_estimator=stl_estimator,
        )
    elif vi_family == "frvi":
        vi_algo = frvi_top_level_api(
            logdensity_fn,
            optimizer,
            num_samples=100,
            objective=objective,
            stl_estimator=stl_estimator,
        )
    else:
        raise ValueError(f"Unknown vi_family: {vi_family}")
    
    key = random.PRNGKey(0)
    state = vi_algo.init(jnp.array([0.0, 0.0, 0.0]))
    
    # optimization
    print(f"   Training {vi_family.upper()}...")
    for i in range(1000):
        key, subkey = random.split(key)
        state, info = vi_algo.step(subkey, state)
        if (i + 1) % 500 == 0:
            objective_value = getattr(info, "objective_value", getattr(info, "elbo", None))
            print(f"      Step {i+1:4d}, objective = {float(objective_value):.4f}")
    
    # get samples
    key, subkey = random.split(key)
    vi_samples = np.array(vi_algo.sample(subkey, state, 10_000))
    
    return vi_samples

# Train both models
print("\n1.3 Training Full Rank VI...")
frvi_samples = run_vi_experiment("frvi", logdensity_fn, low, high, objective, stl_estimator)

print("\n1.4 Training Mean Field VI...")
mfvi_samples = run_vi_experiment("mfvi", logdensity_fn, low, high, objective, stl_estimator)

print("\n Model training complete!\n")


...
```




## 5. Numerical diagnostics

It is time to compare true vs posterior samples mean and variances. Moreover, KL-divergence metric is used to compare the results. The lower the KL metric the better the variational approximation is.




```{code-cell} python3



print("="*100)
print("DIAGNOSTICS: MEANS, VARIANCES, AND KL DIVERGENCE")
print("="*100)

# Function to compute metrics
def compute_metrics(vi_samples, true_samples):
    """Compute statistics and KL divergence."""
    vi_samples = np.array(vi_samples)
    true_samples = np.array(true_samples)
    
    vi_mean = vi_samples.mean(axis=0)
    vi_var = vi_samples.var(axis=0)
    true_mean = true_samples.mean(axis=0)
    true_var = true_samples.var(axis=0)
    
    # KL divergence for diagonal Gaussians
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
    
    return {
        'vi_mean': vi_mean,
        'vi_var': vi_var,
        'true_mean': true_mean,
        'true_var': true_var,
        'kl_divergence': kl_div
    }

# Compute metrics for both models
frvi_metrics = compute_metrics(frvi_samples, true_samples_raw)
mfvi_metrics = compute_metrics(mfvi_samples, true_samples_raw)

# Print means table
print("\n TABLE 1: MEANS COMPARISON")
print("-" * 100)
print(f"{'Variable':<12} {'True Mean':<18} {'Full Rank VI Mean':<22} {'Mean Field VI Mean':<22}")
print("-" * 100)
for i in range(3):
    print(f"x{i:<11} {frvi_metrics['true_mean'][i]:<18.6f} "
          f"{frvi_metrics['vi_mean'][i]:<22.6f} "
          f"{mfvi_metrics['vi_mean'][i]:<22.6f}")

# Print variances table
print("\n TABLE 2: VARIANCES COMPARISON")
print("-" * 100)
print(f"{'Variable':<12} {'True Variance':<18} {'Full Rank VI Var':<22} {'Mean Field VI Var':<22}")
print("-" * 100)
for i in range(3):
    print(f"x{i:<11} {frvi_metrics['true_var'][i]:<18.6f} "
          f"{frvi_metrics['vi_var'][i]:<22.6f} "
          f"{mfvi_metrics['vi_var'][i]:<22.6f}")

# Print KL divergences
print("\n TABLE 3: KL DIVERGENCE COMPARISON")
print("-" * 100)
print(f"{'Method':<25} {'KL Divergence':<20}")
print("-" * 100)
print(f"{'Full Rank VI':<25} {frvi_metrics['kl_divergence']:<20.8f}")
print(f"{'Mean Field VI':<25} {mfvi_metrics['kl_divergence']:<20.8f}")

# Performance comparison
print("\n PERFORMANCE SUMMARY")
print("-" * 100)
if frvi_metrics['kl_divergence'] < mfvi_metrics['kl_divergence']:
    improvement = mfvi_metrics['kl_divergence'] - frvi_metrics['kl_divergence']
    print(f"Full Rank VI performs better")
    print(f"Lower KL divergence by {improvement:.8f}")
else:
    improvement = frvi_metrics['kl_divergence'] - mfvi_metrics['kl_divergence']
    print(f"Mean Field VI performs better")
    print(f"Lower KL divergence by {improvement:.8f}")

print("\n" + "="*100)



...
```








## 6. Visual diagnostics

A corner plot is a nice tool to visualize results while sampling in high-dimensional parameter space



```{code-cell} python3


print("GENERATING CORNER PLOT")
print("="*100)

# Create figure for corner plot
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Plot true Gaussian (red)
corner.corner(true_samples_raw, color="red", label="True Gaussian", 
              hist_kwargs={"density": True, "alpha": 0.6, "linewidth": 1.5},
              fig=fig, axes=axes, show_titles=True, title_fmt=".2f",
              truth_color="red", truth_linewidth=2)

# Plot Full Rank VI (blue)
corner.corner(frvi_samples, color="blue", label="Full Rank VI", 
              hist_kwargs={"density": True, "alpha": 0.6, "linewidth": 1.5},
              fig=fig, axes=axes)

# Plot Mean Field VI (gold)
corner.corner(mfvi_samples, color="green", label="Mean Field VI", 
              hist_kwargs={"density": True, "alpha": 0.6, "linewidth": 1.5},
              fig=fig, axes=axes)

# Add legend
handles = [
    plt.Line2D([], [], color="red", linewidth=2, label="True Gaussian"),
    plt.Line2D([], [], color="blue", linewidth=2, label="Full Rank VI"),
    plt.Line2D([], [], color="green", linewidth=2, label="Mean Field VI"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12)

# Add title
fig.suptitle(f'Distribution Comparison', fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

print("\nCorner plot displayed successfully!")
print("="*100)
print("EXPERIMENT COMPLETE")
print("="*100)



...
```

