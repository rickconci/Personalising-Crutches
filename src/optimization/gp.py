"""Hand-written Gaussian Process regression (numpy + scipy).

Mirrors the structure of the L48 reference notebooks:
    * Stationary kernels (RBF and Matern-5/2) with a single isotropic
      lengthscale, signal variance, and observation noise variance.
    * Cholesky-stable posterior mean/covariance with a small jitter fallback
      when the Gram matrix is poorly conditioned.
    * Negative log marginal likelihood + L-BFGS-B fit over log-thetas with
      multi-start restarts.

Type aliases:
    * ``Theta = (signal_variance, lengthscale, noise_variance)``
    * ``KernelFn = Callable[[X1, X2 | None, Theta], np.ndarray]``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


KernelName = Literal["rbf", "matern52"]

#: ``theta = (signal_variance, lengthscale, noise_variance)`` (all positive).
Theta = np.ndarray  # shape (3,)
KernelFn = Callable[[np.ndarray, np.ndarray | None, Theta], np.ndarray]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _pairwise_distance(x1: np.ndarray, x2: np.ndarray | None) -> np.ndarray:
    """Pairwise Euclidean distance ``cdist(x1, x2 or x1)``."""
    return cdist(x1, x1) if x2 is None else cdist(x1, x2)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray | None, theta: Theta) -> np.ndarray:
    """Squared-exponential kernel.

    ``k(x, x') = signal_variance * exp(- d(x, x')^2 / (2 * lengthscale^2))``.

    Args:
        x1: ``(n, d)`` array.
        x2: ``(m, d)`` array or ``None`` (means ``x2 == x1``).
        theta: ``(signal_variance, lengthscale, noise_variance)``.

    Returns:
        ``(n, m)`` covariance matrix (noise is *not* added here -- see
        :func:`gp_posterior`).
    """
    signal_var, lengthscale, _ = theta
    d = _pairwise_distance(x1, x2)
    return signal_var * np.exp(-0.5 * (d / lengthscale) ** 2)


def matern52_kernel(x1: np.ndarray, x2: np.ndarray | None, theta: Theta) -> np.ndarray:
    """Matern-5/2 kernel (twice-differentiable; popular default for BO)."""
    signal_var, lengthscale, _ = theta
    d = _pairwise_distance(x1, x2)
    sqrt5_d_l = np.sqrt(5.0) * d / lengthscale
    return signal_var * (1.0 + sqrt5_d_l + (5.0 / 3.0) * (d / lengthscale) ** 2) * np.exp(-sqrt5_d_l)


KERNELS: dict[KernelName, KernelFn] = {"rbf": rbf_kernel, "matern52": matern52_kernel}


def get_kernel(name: KernelName) -> KernelFn:
    """Lookup a registered kernel by name (raises ``ValueError`` if unknown)."""
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel {name!r}; expected one of {list(KERNELS)}")
    return KERNELS[name]


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------
def _cholesky_with_jitter(
    K: np.ndarray, base_jitter: float = 1e-6, max_jitter: float = 1.0
) -> tuple[np.ndarray, float]:
    """Cholesky factorisation with progressive jitter on failure.

    Returns the lower-triangular ``L`` and the jitter that was applied. Raises
    ``np.linalg.LinAlgError`` only if even ``max_jitter`` cannot stabilise.
    """
    n = K.shape[0]
    jitter = base_jitter
    while jitter <= max_jitter:
        try:
            return np.linalg.cholesky(K + jitter * np.eye(n)), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(
        f"Cholesky failed even with jitter={max_jitter}"
    )


@dataclass(slots=True)
class GPPosterior:
    """Posterior at the test inputs.

    Attributes:
        mean: ``(n_test,)`` predictive mean.
        cov: ``(n_test, n_test)`` predictive covariance (no obs noise added).
        var: ``(n_test,)`` predictive variance (diagonal of ``cov``, clipped
            non-negative for numerical safety).
    """

    mean: np.ndarray
    cov: np.ndarray
    var: np.ndarray


def gp_posterior(
    x_star: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    theta: Theta,
    kernel: KernelFn,
) -> GPPosterior:
    """Predictive distribution at ``x_star`` given training data.

    Args:
        x_star: ``(n_test, d)`` test inputs.
        x_train: ``(n_train, d)`` training inputs.
        y_train: ``(n_train,)`` training targets (assumed centred -- caller
            should subtract the prior mean if needed).
        theta: ``(signal_variance, lengthscale, noise_variance)``.
        kernel: A kernel function from :data:`KERNELS`.

    Returns:
        :class:`GPPosterior`.
    """
    if x_train.shape[0] == 0:
        # Prior predictive
        K_ss = kernel(x_star, None, theta)
        var = np.clip(np.diag(K_ss), 0.0, None)
        return GPPosterior(mean=np.zeros(x_star.shape[0]), cov=K_ss, var=var)

    noise = max(float(theta[2]), 1e-12)
    K = kernel(x_train, None, theta) + noise * np.eye(x_train.shape[0])
    K_s = kernel(x_train, x_star, theta)
    K_ss = kernel(x_star, None, theta)

    L, _ = _cholesky_with_jitter(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    cov = K_ss - v.T @ v
    var = np.clip(np.diag(cov), 0.0, None)
    return GPPosterior(mean=mean, cov=cov, var=var)


# ---------------------------------------------------------------------------
# Marginal likelihood + hyperparameter fit
# ---------------------------------------------------------------------------
def negative_log_marginal_likelihood(
    theta: Theta,
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: KernelFn,
) -> float:
    """Standard GP NLL (per Rasmussen & Williams Eq. 2.30).

    Returns ``+inf`` if the kernel matrix is so degenerate that even the
    jitter-fallback Cholesky cannot factorise it -- this lets ``minimize``
    treat it as an out-of-domain proposal.
    """
    n = x_train.shape[0]
    noise = max(float(theta[2]), 1e-12)
    try:
        K = kernel(x_train, None, theta) + noise * np.eye(n)
        L, _ = _cholesky_with_jitter(K)
    except np.linalg.LinAlgError:
        return float("inf")

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    nll = 0.5 * float(y_train @ alpha)
    nll += float(np.sum(np.log(np.diag(L))))
    nll += 0.5 * n * np.log(2.0 * np.pi)
    return nll


@dataclass(slots=True)
class GPModel:
    """Fitted GP: kernel + theta + cached training data.

    Use :meth:`predict` to obtain :class:`GPPosterior` at new inputs.
    """

    kernel_name: KernelName
    theta: Theta
    x_train: np.ndarray
    y_train: np.ndarray
    train_nll: float

    @property
    def kernel(self) -> KernelFn:
        return get_kernel(self.kernel_name)

    def predict(self, x_star: np.ndarray) -> GPPosterior:
        return gp_posterior(x_star, self.x_train, self.y_train, self.theta, self.kernel)


def _theta_from_log(log_theta: np.ndarray) -> Theta:
    return np.exp(log_theta)


def fit_gp_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kernel: KernelName = "matern52",
    n_restarts: int = 5,
    init_log_theta: np.ndarray | None = None,
    seed: int | None = 0,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> GPModel:
    """Fit ``theta`` by maximising the marginal likelihood (L-BFGS-B over log-theta).

    Optimisation is in log-space so we can use unconstrained gradients while
    enforcing positivity. We restart from random points in
    ``log_bounds`` and keep the best NLL.

    Args:
        x_train: ``(n, d)`` training inputs (z-scored upstream is recommended).
        y_train: ``(n,)`` training targets (centred recommended).
        kernel: ``"rbf"`` or ``"matern52"``.
        n_restarts: Number of random restarts (in addition to ``init_log_theta``).
        init_log_theta: Optional explicit starting point in log-space.
        seed: RNG seed for the restarts.
        bounds: Optional log-space bounds per dim; defaults to a wide box.

    Returns:
        :class:`GPModel` with the best ``theta`` and cached training data.
    """
    if x_train.shape[0] == 0:
        raise ValueError("Cannot fit GP with zero training points")
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("x_train / y_train length mismatch")

    kernel_fn = get_kernel(kernel)
    if bounds is None:
        bounds = ((np.log(1e-3), np.log(1e3)),) * 3  # signal_var, lengthscale, noise_var

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    if init_log_theta is not None:
        starts.append(np.asarray(init_log_theta, dtype=float))
    starts.append(np.array([0.0, 0.0, np.log(0.1)]))  # signal=1, length=1, noise=0.1
    for _ in range(max(0, n_restarts)):
        starts.append(
            np.array(
                [
                    rng.uniform(np.log(0.1), np.log(10.0)),
                    rng.uniform(np.log(0.3), np.log(10.0)),
                    rng.uniform(np.log(1e-3), np.log(1.0)),
                ]
            )
        )

    best_theta: Theta | None = None
    best_nll = float("inf")

    for log_theta0 in starts:
        try:
            res = minimize(
                fun=lambda lt: negative_log_marginal_likelihood(
                    _theta_from_log(lt), x_train, y_train, kernel_fn
                ),
                x0=np.asarray(log_theta0, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=200, ftol=1e-9),
            )
        except Exception as e:  # noqa: BLE001 -- best-effort restart
            logger.debug("Restart failed: %s", e)
            continue
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta = _theta_from_log(res.x)

    if best_theta is None:
        raise RuntimeError("All hyperparameter restarts failed")

    return GPModel(
        kernel_name=kernel,
        theta=best_theta,
        x_train=x_train,
        y_train=y_train,
        train_nll=best_nll,
    )
