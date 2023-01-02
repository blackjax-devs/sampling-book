from typing import Callable, Dict, Tuple

import aeppl
import aesara
import aesara.tensor as at
import jax
import numpy as np
import pandas as pd
from aeppl.transforms import LogTransform, TransformValuesRewrite
from aesara.tensor.random import RandomVariable

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
)


def model(
    X: np.ndarray,
) -> Tuple[RandomVariable, RandomVariable, RandomVariable, RandomVariable]:
    """Build the sparse regression model."""
    X_at = at.as_tensor(X)

    srng = at.random.RandomStream(0)

    tau_rv = srng.halfcauchy(0, 1)
    lambda_rv = srng.halfcauchy(0, 1, size=X_at.shape[-1])

    theta = tau_rv * lambda_rv
    beta_rv = srng.normal(0, theta, size=X_at.shape[-1])

    eta = X_at @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p)

    return tau_rv, lambda_rv, beta_rv, Y_rv


def german_credit_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Download and prepare the German Credit dataset."""

    data = pd.read_table(DATA_URL, header=None, delim_whitespace=True)

    # In the dataset, "1" indicates a customer with good credit, and "2"
    # a customer with bad credit. We adjust this to perform a Bernoulli
    # regression.
    y = -1 * (data.iloc[:, -1].values - 2)

    # We normalize the inputs to the regression.
    X = (
        data.iloc[:, :-1]
        .apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0)
        .values
    )
    X = np.concatenate([np.ones((1000, 1)), X], axis=1)

    return X, y


def logdensity() -> Callable[[Dict], jax.Array]:
    """Return a function that computes the log-density of the model."""
    X, y = german_credit_dataset()
    tau_rv, lambda_rv, beta_rv, Y_rv = model(X)

    transforms_op = TransformValuesRewrite(
        {lambda_rv: LogTransform(), tau_rv: LogTransform()}
    )

    logdensity, value_variables = aeppl.joint_logprob(
        tau_rv,
        lambda_rv,
        beta_rv,
        realized={Y_rv: at.as_tensor(y)},
        extra_rewrites=transforms_op,
    )

    logdensity_aesara_fn = aesara.function(
        list(value_variables), logdensity, mode="JAX"
    )

    def logdensity_fn(position: Dict) -> jax.Array:
        """Computes the model's logdensity.

        Assumes that the position is passed as a dictionary.
        """
        tau = position["log_tau"]
        lmbda = position["log_lambda"]
        beta = position["beta"]
        return logdensity_aesara_fn.vm.jit_fn(tau, lmbda, beta)[0]

    return logdensity_fn
