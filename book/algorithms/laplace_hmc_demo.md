---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: blackjax
  language: python
  name: blackjax
---

# Adjoint-Differentiated Laplace HMC

This notebook demonstrates `blackjax.laplace_hmc`, a sampler that integrates out
latent Gaussian variables via the Laplace approximation and runs HMC over the
marginal hyperparameter posterior.

**Core idea.** In a hierarchical model
```
phi   ~ p(phi)             # hyperparameters  (low-dimensional)
theta ~ N(0, K(phi))       # latent variables  (high-dimensional)
y     ~ p(y | theta, phi)  # likelihood
```
sampling the joint $(\theta, \phi)$ posterior is hard because the geometry changes
dramatically with $\phi$ ("funnel" geometry). Laplace-HMC instead runs HMC over
$\phi$ alone, using the Laplace approximation to analytically handle $\theta$. The
mode $\theta^*(\phi) = \text{argmax}_\theta \log p(\theta | \phi, y)$ is found via L-BFGS, and
the log-marginal gradient uses the implicit function theorem — no gradient
unrolling through the optimizer.

**This notebook demonstrates:**
1. **Efficiency on Neal's Funnel** — where Laplace HMC excels by marginalizing away the difficult geometry.
2. **Vector Hyperparameters and Warmup** — using a multi-group model and automatic adaptation.

```{code-cell} ipython3
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree

import blackjax
import blackjax.mcmc.laplace_marginal
from blackjax.optimizers.lbfgs import minimize_lbfgs

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

# Reproducibility
from datetime import date
key = jax.random.key(int(date.today().strftime("%Y%m%d")))

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    return jax.lax.scan(one_step, initial_state, keys)
```

## Section 1: Neal's Funnel (Speed and Efficiency)

Neal's Funnel is a classic example of a "difficult" geometry for MCMC. The joint distribution $p(\theta, v)$ is:
$$
\begin{aligned}
v &\sim N(0, 3^2) \\
\theta_i &\sim N(0, \exp(v)^2) \quad \text{for } i=1 \dots n
\end{aligned}
$$
In the joint space, as $v$ becomes small, the conditional distribution of $\theta$ becomes very narrow, forcing HMC to take tiny steps. Laplace HMC marginalizes out $\theta$ analytically (the approximation is exact here!), leaving a simple Gaussian for $v$.

```{code-cell} ipython3
n_theta = 100

def funnel_model(n_obs=100):
    v = numpyro.sample("v", dist.Normal(0.0, 3.0))
    with numpyro.plate("data", n_obs):
        numpyro.sample("theta", dist.Normal(0.0, jnp.exp(v)))

init_key, key = jax.random.split(key)
_, potential_fn, _, _ = initialize_model(init_key, funnel_model, model_args=(n_theta,))

def laplace_log_joint(theta, v):
    return -potential_fn({'theta': theta, 'v': v})

def nuts_log_joint(params):
    return -potential_fn(params)
```

### Laplace HMC vs NUTS

```{code-cell} ipython3
# 1. Setup Laplace Marginal
laplace_funnel = blackjax.mcmc.laplace_marginal.laplace_marginal_factory(
    laplace_log_joint, 
    theta_init=jnp.zeros(n_theta),
)

# 2. Setup and run Window Adaptation for Laplace-HMC
start = time.time()
print("Running Laplace-HMC window adaptation on Funnel...")
warmup_laplace = blackjax.window_adaptation(
    blackjax.laplace_hmc,
    laplace_funnel,
    num_integration_steps=3,  # 3 leapfrog steps: each requires an L-BFGS solve,
                               # so fewer steps = much faster; 1D marginal mixes fine
)

key, warmup_key = jax.random.split(key)
(state_laplace, parameters_laplace), _ = warmup_laplace.run(warmup_key, jnp.array([0.0]), num_steps=500)

# 3. Sampling
sampler_laplace = blackjax.laplace_hmc(
    laplace_log_joint, 
    theta_init=jnp.zeros(n_theta), 
    **parameters_laplace
)

print("Running Laplace-HMC sampling on Funnel...")
key, sample_key = jax.random.split(key)
_, (states_laplace, _) = inference_loop(sample_key, sampler_laplace.step, state_laplace, 2000)
states_laplace.position.block_until_ready()
laplace_time = time.time() - start

# NUTS
start = time.time()
warmup = blackjax.window_adaptation(blackjax.nuts, nuts_log_joint)
key, warmup_key = jax.random.split(key)
(state_nuts, parameters_nuts), _ = warmup.run(warmup_key, {'v': jnp.array([0.0]), 'theta': jnp.zeros(n_theta)}, num_steps=1000)
nuts_sampler = blackjax.nuts(nuts_log_joint, **parameters_nuts)

print("Running NUTS on Funnel...")
key, sample_key = jax.random.split(key)
_, (states_nuts, _) = inference_loop(sample_key, nuts_sampler.step, state_nuts, 2000)
states_nuts.position['v'].block_until_ready()
nuts_time = time.time() - start

print(f"Laplace-HMC time: {laplace_time:.2f}s")
print(f"NUTS time:        {nuts_time:.2f}s")
```

```{code-cell} ipython3
from blackjax.diagnostics import effective_sample_size

fig, ax = plt.subplots(figsize=(8, 4))
v_laplace = states_laplace.position[1000:]
v_nuts = states_nuts.position['v'][1000:]

# Compute actual ESS (chains=1, draws, params=1)
ess_laplace = float(effective_sample_size(v_laplace[None, :, None]))
ess_nuts    = float(effective_sample_size(v_nuts[None, :, None]))

ax.hist(np.array(v_laplace), bins=30, density=True, alpha=0.6, color="steelblue",
        label=f"Laplace-HMC  ESS/s={ess_laplace/laplace_time:.0f}  ({laplace_time:.1f}s)")
ax.hist(np.array(v_nuts), bins=30, density=True, alpha=0.5, color="orange",
        label=f"NUTS  ESS/s={ess_nuts/nuts_time:.0f}  ({nuts_time:.1f}s)")
# Exact marginal for v is N(0, 3^2)
v_range = jnp.linspace(-10, 10, 100)
ax.plot(v_range, jax.scipy.stats.norm.pdf(v_range, 0, 3), 'r--', label="Exact Marginal")
ax.set_title("Neal's Funnel: Marginal of v")
ax.legend()
plt.show()
```

## Section 2: Vector Hyperparameters and Warmup

In many cases, we have multiple hyperparameters. Here we use a model with two groups,
each having its own log-sigma hyperparameter $\phi_1, \phi_2$. To ensure the
hyperparameters are identifiable, we provide multiple observations per latent variable
$\theta_i$.

```{code-cell} ipython3
import seaborn as sns

n_groups = 2
n_latents_per_group = 20
n_obs_per_latent = 10

# --- Data generation ---
key, k1, k2, k3 = jax.random.split(key, 4)
phi_true = jnp.array([0.5, -0.5])
sigma_true = jnp.exp(phi_true)

theta_true = jnp.concatenate([
    jax.random.normal(k1, (n_latents_per_group,)) * sigma_true[0],
    jax.random.normal(k2, (n_latents_per_group,)) * sigma_true[1]
])

y_prob = jax.nn.sigmoid(theta_true[:, None])
y_obs = (
    jax.random.uniform(k3, (n_groups * n_latents_per_group, n_obs_per_latent)) < y_prob
).astype(jnp.float32)
# Reshape for NumPyro plates: (obs_per_latent, groups, latents_per_group)
y_obs = y_obs.T.reshape(n_obs_per_latent, n_groups, n_latents_per_group)

# --- Model ---
def multi_group_model(y=None):
    phi = numpyro.sample("phi", dist.Normal(jnp.zeros(n_groups), 2.0))
    sigma = jnp.exp(phi)[:, None]
    with numpyro.plate("groups", n_groups, dim=-2):
        with numpyro.plate("latents", n_latents_per_group, dim=-1):
            theta = numpyro.sample("theta", dist.Normal(0.0, sigma))
            with numpyro.plate("observations", n_obs_per_latent, dim=-3):
                numpyro.sample("y", dist.Bernoulli(logits=theta), obs=y)

init_key, key = jax.random.split(key)
_, potential_fn_vec, _, _ = initialize_model(
    init_key, multi_group_model, model_args=(y_obs,)
)

def log_joint_vec(theta, phi):
    return -potential_fn_vec({"theta": theta, "phi": phi})

laplace_vec = blackjax.mcmc.laplace_marginal.laplace_marginal_factory(
    log_joint_vec,
    theta_init=jnp.zeros((n_groups, n_latents_per_group)),
    maxiter=100,
)
```

```{code-cell} ipython3
# --- Laplace-HMC: warmup + sampling ---
warmup = blackjax.window_adaptation(
    blackjax.laplace_hmc, laplace_vec, num_integration_steps=12
)
key, warmup_key = jax.random.split(key)

start = time.time()
(state_vec, parameters_vec), _ = warmup.run(warmup_key, jnp.zeros(n_groups), num_steps=500)
sampler_vec = blackjax.laplace_hmc(
    log_joint_vec,
    theta_init=jnp.zeros((n_groups, n_latents_per_group)),
    **parameters_vec,
)
key, sample_key = jax.random.split(key)
_, (states_vec, _) = inference_loop(sample_key, sampler_vec.step, state_vec, 1000)
states_vec.position.block_until_ready()
laplace_vec_time = time.time() - start

phi_samples = states_vec.position[500:]
theta_star_samples = states_vec.theta_star[500:]

# Recover theta ~ N(theta_star, H^{-1}) for each phi sample
key, rk = jax.random.split(key)
theta_samples_laplace = jax.vmap(laplace_vec.sample_theta)(
    jax.random.split(rk, len(phi_samples)), phi_samples, theta_star_samples
)

print(f"Laplace-HMC  total: {laplace_vec_time:.1f}s  "
      f"(warmup 500 + sampling 1000, phi dim={n_groups})")
print(f"phi_1 mean: {float(phi_samples[:, 0].mean()):.3f}  (true {float(phi_true[0]):.1f})")
print(f"phi_2 mean: {float(phi_samples[:, 1].mean()):.3f}  (true {float(phi_true[1]):.1f})")
print(f"theta shape: {theta_samples_laplace.shape}")
```

```{code-cell} ipython3
# --- NUTS on the full joint (phi, theta) ---
def nuts_log_joint_vec(params):
    return -potential_fn_vec(params)

warmup_nuts = blackjax.window_adaptation(blackjax.nuts, nuts_log_joint_vec)
key, warmup_key = jax.random.split(key)

start = time.time()
(state_nuts, parameters_nuts), _ = warmup_nuts.run(
    warmup_key,
    {"phi": jnp.zeros(n_groups), "theta": jnp.zeros((n_groups, n_latents_per_group))},
    num_steps=1000,
)
nuts_sampler = blackjax.nuts(nuts_log_joint_vec, **parameters_nuts)
key, sample_key = jax.random.split(key)
_, (states_nuts, _) = inference_loop(sample_key, nuts_sampler.step, state_nuts, 1000)
states_nuts.position["phi"].block_until_ready()
nuts_vec_time = time.time() - start

phi_nuts = states_nuts.position["phi"][500:]
theta_nuts = states_nuts.position["theta"][500:]
print(f"NUTS  total: {nuts_vec_time:.1f}s  "
      f"(warmup 1000 + sampling 1000, joint dim={n_groups + n_groups*n_latents_per_group})")
print(f"phi_1 mean: {float(phi_nuts[:, 0].mean()):.3f}")
print(f"phi_2 mean: {float(phi_nuts[:, 1].mean()):.3f}")
```

```{code-cell} ipython3
:tags: [hide-cell]
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 6))
gs_outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Left: jointplot-style density for phi posterior ---
gs_joint = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=gs_outer[0], hspace=0.05, wspace=0.05
)
ax_joint = fig.add_subplot(gs_joint[1:, :-1])
ax_top   = fig.add_subplot(gs_joint[0, :-1], sharex=ax_joint)
ax_right = fig.add_subplot(gs_joint[1:, -1], sharey=ax_joint)

for phi_arr, color, label in [
    (phi_samples, "steelblue", "Laplace-HMC"),
    (phi_nuts,    "orange",    "NUTS"),
]:
    x, y = np.array(phi_arr[:, 0]), np.array(phi_arr[:, 1])
    sns.kdeplot(x=x, y=y, fill=True, alpha=0.45, color=color, ax=ax_joint)
    sns.kdeplot(x=x, fill=True, alpha=0.5, color=color, ax=ax_top, label=label)
    sns.kdeplot(y=y, fill=True, alpha=0.5, color=color, ax=ax_right)

ax_joint.axvline(float(phi_true[0]), color="r", ls="--", lw=1.2, label="True φ")
ax_joint.axhline(float(phi_true[1]), color="r", ls="--", lw=1.2)
ax_joint.set_xlabel("φ₁ (log σ₁)")
ax_joint.set_ylabel("φ₂ (log σ₂)")

plt.setp(ax_top.get_xticklabels(), visible=False)
plt.setp(ax_right.get_yticklabels(), visible=False)
ax_top.set_ylabel("")
ax_right.set_xlabel("")
ax_top.legend(fontsize=8, loc="upper left")
ax_joint.legend(fontsize=8, loc="lower right")
ax_top.set_title("Posterior of φ = (log σ₁, log σ₂)")

# --- Right: latent variable posterior mean ± 2σ ---
ax2 = fig.add_subplot(gs_outer[1])
group_idx = 0
x_idx = jnp.arange(n_latents_per_group)
for mean, std, fmt, color, label in [
    (theta_samples_laplace[:, group_idx, :].mean(0),
     theta_samples_laplace[:, group_idx, :].std(0),
     "o", "steelblue", "Laplace-HMC"),
    (theta_nuts[:, group_idx, :].mean(0),
     theta_nuts[:, group_idx, :].std(0),
     "x", "orange", "NUTS"),
]:
    ax2.errorbar(
        np.array(x_idx + (0.15 if fmt == "x" else -0.15)),
        np.array(mean), yerr=2 * np.array(std),
        fmt=fmt, color=color, alpha=0.7, markersize=4, label=label,
    )
ax2.set_xlabel("Latent index i")
ax2.set_ylabel("θᵢ")
ax2.set_title(f"Latent variables — Group {group_idx + 1}")
ax2.legend(fontsize=9)

plt.show()
```

## Section 3: Gaussian Process Regression (Laplace is Exact)

In a GP regression model with Gaussian likelihood the Laplace approximation is
**exact**: the conditional posterior $p(\theta \mid \phi, y)$ is itself Gaussian, so
the marginal $\log p(\phi \mid y)$ is computed without any approximation error.
Laplace-HMC samples the two kernel hyperparameters $\phi = (\log\ell, \log\sigma_f)$;
the $n$-dimensional function values $\theta$ are recovered afterwards via
`laplace.sample_theta`.

$$
\phi_j \sim \mathcal{N}(0, 1), \qquad
\theta \mid \phi \sim \mathcal{N}(0, K_\phi), \qquad
y_i \mid \theta_i \sim \mathcal{N}(\theta_i, \sigma_n^2)
$$

where $K_\phi$ is the RBF (squared-exponential) kernel.  NUTS must explore the
full $(n+2)$-dimensional space $(\theta, \phi)$ whose geometry is funnel-shaped;
Laplace-HMC only explores the 2-dimensional $\phi$ space.

```{code-cell} ipython3
# --- Data -----------------------------------------------------------------
n_gp = 30                                         # 30 training points
X_gp = jnp.linspace(0, 8, n_gp)
f_true = jnp.sin(X_gp)
sigma_n_gp = 0.3
key, k_data = jax.random.split(key)
y_gp = f_true + sigma_n_gp * jax.random.normal(k_data, (n_gp,))

# --- RBF kernel -----------------------------------------------------------
def rbf_kernel(x1, x2, log_ell, log_sigma_f):
    ell, sigma_f = jnp.exp(log_ell), jnp.exp(log_sigma_f)
    r2 = ((x1[:, None] - x2[None, :]) / ell) ** 2
    return sigma_f ** 2 * jnp.exp(-0.5 * r2)

def log_joint_gp(theta, phi):
    log_ell, log_sigma_f = phi[0], phi[1]
    K = rbf_kernel(X_gp, X_gp, log_ell, log_sigma_f) + 1e-5 * jnp.eye(n_gp)
    log_p_phi = (
        jax.scipy.stats.norm.logpdf(log_ell, 0.0, 1.0)
        + jax.scipy.stats.norm.logpdf(log_sigma_f, 0.0, 1.0)
    )
    log_p_theta = jax.scipy.stats.multivariate_normal.logpdf(
        theta, jnp.zeros(n_gp), K
    )
    log_lik = jax.scipy.stats.norm.logpdf(y_gp, theta, sigma_n_gp).sum()
    return log_p_phi + log_p_theta + log_lik
```

```{code-cell} ipython3
# --- Laplace-HMC ----------------------------------------------------------
laplace_gp = blackjax.mcmc.laplace_marginal.laplace_marginal_factory(
    log_joint_gp, jnp.zeros(n_gp), maxiter=100
)
warmup_gp = blackjax.window_adaptation(
    blackjax.laplace_hmc, laplace_gp, num_integration_steps=5
)
key, warmup_key = jax.random.split(key)
(state_gp, params_gp), _ = warmup_gp.run(warmup_key, jnp.zeros(2), num_steps=500)

sampler_gp = blackjax.laplace_hmc(log_joint_gp, theta_init=jnp.zeros(n_gp), **params_gp)

print("Sampling GP hyperparameters with Laplace-HMC ...")
key, sample_key = jax.random.split(key)
start = time.time()
_, (states_gp, _) = inference_loop(sample_key, sampler_gp.step, state_gp, 1000)
states_gp.position.block_until_ready()
laplace_gp_time = time.time() - start
print(f"  {laplace_gp_time:.1f}s  (2-D phi space)")

# Recover function values theta ~ N(theta_star, H^{-1}) for each phi sample
phi_gp_samples    = states_gp.position[500:]
theta_star_samples = states_gp.theta_star[500:]
key, rk = jax.random.split(key)
theta_gp_samples = jax.vmap(laplace_gp.sample_theta)(
    jax.random.split(rk, len(phi_gp_samples)), phi_gp_samples, theta_star_samples
)
print(f"  theta shape: {theta_gp_samples.shape}")
```

```{code-cell} ipython3
# --- NUTS on full joint (theta, phi) for comparison ----------------------
def nuts_log_joint_gp(params):
    return log_joint_gp(params["theta"], params["phi"])

warmup_nuts_gp = blackjax.window_adaptation(blackjax.nuts, nuts_log_joint_gp)
key, warmup_key = jax.random.split(key)
(state_nuts_gp, params_nuts_gp), _ = warmup_nuts_gp.run(
    warmup_key,
    {"theta": jnp.zeros(n_gp), "phi": jnp.zeros(2)},
    num_steps=500,
)
nuts_sampler_gp = blackjax.nuts(nuts_log_joint_gp, **params_nuts_gp)

n_nuts_gp = 500
print(f"Sampling full ({n_gp+2}-D) joint with NUTS ({n_nuts_gp} samples) ...")
key, sample_key = jax.random.split(key)
start = time.time()
_, (states_nuts_gp, _) = inference_loop(
    sample_key, nuts_sampler_gp.step, state_nuts_gp, n_nuts_gp
)
states_nuts_gp.position["theta"].block_until_ready()
nuts_gp_time = time.time() - start
print(f"  {nuts_gp_time:.1f}s  ({n_gp+2}-D joint space)")
```

```{code-cell} ipython3
theta_gp_mean = theta_gp_samples.mean(axis=0)
theta_gp_std  = theta_gp_samples.std(axis=0)

nuts_burnin = n_nuts_gp // 2
theta_nuts_mean = states_nuts_gp.position["theta"][nuts_burnin:].mean(axis=0)
theta_nuts_std  = states_nuts_gp.position["theta"][nuts_burnin:].std(axis=0)
phi_nuts_gp     = states_nuts_gp.position["phi"][nuts_burnin:]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: GP posterior at training points
ax = axes[0]
ax.scatter(np.array(X_gp), np.array(y_gp), s=20, alpha=0.5, color="gray", zorder=5, label="Data")
ax.plot(np.array(X_gp), np.array(f_true), "k--", lw=1.5, label="True f", zorder=4)
for mean, std, color, lbl in [
    (theta_gp_mean, theta_gp_std, "steelblue", f"Laplace-HMC ({laplace_gp_time:.0f}s)"),
    (theta_nuts_mean, theta_nuts_std, "orange",
     f"NUTS on (θ,φ) ({n_nuts_gp} samples, {nuts_gp_time:.0f}s)"),
]:
    ax.plot(np.array(X_gp), np.array(mean), color=color, lw=2, label=lbl)
    ax.fill_between(
        np.array(X_gp),
        np.array(mean - 2 * std),
        np.array(mean + 2 * std),
        alpha=0.2, color=color,
    )
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("GP Posterior (mean ± 2σ)")
ax.legend(fontsize=9)

# Right: kernel hyperparameter posterior
ax = axes[1]
ax.scatter(
    np.array(phi_gp_samples[:, 0]), np.array(phi_gp_samples[:, 1]),
    s=8, alpha=0.3, color="steelblue", label="Laplace-HMC",
)
ax.scatter(
    np.array(phi_nuts_gp[:, 0]), np.array(phi_nuts_gp[:, 1]),
    s=8, alpha=0.3, color="orange", label="NUTS",
)
ax.set_xlabel("log length-scale  (log ℓ)")
ax.set_ylabel("log amplitude  (log σ_f)")
ax.set_title("Kernel Hyperparameter Posterior")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Normalise by sample count for a fair per-sample comparison
time_per_sample_laplace = laplace_gp_time / 1000
time_per_sample_nuts    = nuts_gp_time / n_nuts_gp
speedup = time_per_sample_nuts / time_per_sample_laplace
print(f"Time / sample — Laplace-HMC: {time_per_sample_laplace*1000:.0f} ms  "
      f"NUTS: {time_per_sample_nuts*1000:.0f} ms  ({speedup:.1f}x speedup)")
print(f"log length-scale:  Laplace {float(phi_gp_samples[:,0].mean()):.2f}  "
      f"NUTS {float(phi_nuts_gp[:,0].mean()):.2f}")
print(f"log amplitude:     Laplace {float(phi_gp_samples[:,1].mean()):.2f}  "
      f"NUTS {float(phi_nuts_gp[:,1].mean()):.2f}")
```
