"""Hand-written Gaussian Process regression (numpy + scipy).

Mirrors the structure of the L48 reference notebooks but extended with
Automatic Relevance Determination (ARD):

    * Stationary kernels (RBF and Matern-5/2) supporting either a single
      isotropic lengthscale (``"rbf"`` / ``"matern52"``) or one lengthscale
      per input dimension (``"rbf_ard"`` / ``"matern52_ard"``).
    * Cholesky-stable posterior mean/covariance with a small jitter fallback
      when the Gram matrix is poorly conditioned.
    * Negative log marginal likelihood + L-BFGS-B fit over log-thetas with
      multi-start restarts.

Theta layout:
    * Isotropic kernel: ``theta = (signal_variance, ℓ, noise_variance)``,
      length 3.
    * ARD kernel: ``theta = (signal_variance, ℓ_1, ..., ℓ_d, noise_variance)``,
      length ``2 + d``.

The kernel functions handle both via NumPy broadcasting of ``x / ℓ``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


KernelName = Literal["rbf", "matern52", "rbf_ard", "matern52_ard"]

#: Flat parameter vector. Length is ``2 + n_lengthscales``.
Theta = np.ndarray
KernelFn = Callable[[np.ndarray, np.ndarray | None, Theta], np.ndarray]


# ---------------------------------------------------------------------------
# Kernel name helpers
# ---------------------------------------------------------------------------
def is_ard(name: KernelName) -> bool:
    """Return True when ``name`` selects an Automatic-Relevance-Determination kernel."""
    return name.endswith("_ard")


def base_kernel_name(name: KernelName) -> str:
    """Strip the ``_ard`` suffix to get the underlying kernel family."""
    return name.removesuffix("_ard")


def n_lengthscales(name: KernelName, n_features: int) -> int:
    """Number of lengthscale hyperparameters for ``name`` in ``n_features``-dim inputs."""
    return n_features if is_ard(name) else 1


def n_theta(name: KernelName, n_features: int) -> int:
    """Total length of the parameter vector for ``name`` in ``n_features``-dim inputs."""
    return 2 + n_lengthscales(name, n_features)


# ---------------------------------------------------------------------------
# Kernels (broadcast-aware: ells is shape (1,) for iso or (d,) for ARD)
# ---------------------------------------------------------------------------
def _scaled_distance(
    x1: np.ndarray, x2: np.ndarray | None, ells: np.ndarray
) -> np.ndarray:
    """Pairwise Euclidean distance after per-dim rescaling by ``ells``.

    Equivalently, the Mahalanobis distance with covariance ``diag(ells**2)``.
    For isotropic kernels ``ells.shape == (1,)`` and this reduces to the
    standard ``d(x1, x2) / ℓ``.
    """
    x1s = x1 / ells
    x2s = x1s if x2 is None else x2 / ells
    return cdist(x1s, x2s)


def _split_theta(theta: Theta) -> tuple[float, np.ndarray, float]:
    """Split a flat theta vector into (signal_var, ells, noise_var)."""
    signal_var = float(theta[0])
    noise_var = float(theta[-1])
    ells = np.asarray(theta[1:-1], dtype=float)
    return signal_var, ells, noise_var


def rbf_kernel(x1: np.ndarray, x2: np.ndarray | None, theta: Theta) -> np.ndarray:
    """Squared-exponential kernel (isotropic or ARD).

    ``k(x, x') = signal_variance * exp(- 0.5 * sum_i ((x_i - x'_i) / ℓ_i)**2)``

    Args:
        x1: ``(n, d)`` array.
        x2: ``(m, d)`` array or ``None`` (means ``x2 == x1``).
        theta: Flat ``(2 + n_lengthscales,)`` parameter vector. Layout:
            ``(signal_var, ℓ_1, ..., ℓ_n, noise_var)``. ``n_lengthscales`` is
            ``1`` (isotropic) or ``d`` (ARD).

    Returns:
        ``(n, m)`` covariance matrix (noise is *not* added here -- see
        :func:`gp_posterior`).
    """
    signal_var, ells, _ = _split_theta(theta)
    d = _scaled_distance(x1, x2, ells)
    return signal_var * np.exp(-0.5 * d**2)


def matern52_kernel(x1: np.ndarray, x2: np.ndarray | None, theta: Theta) -> np.ndarray:
    """Matern-5/2 kernel (isotropic or ARD; twice-differentiable BO default)."""
    signal_var, ells, _ = _split_theta(theta)
    d = _scaled_distance(x1, x2, ells)
    sqrt5_d = np.sqrt(5.0) * d
    return signal_var * (1.0 + sqrt5_d + (5.0 / 3.0) * d**2) * np.exp(-sqrt5_d)


# Map *base* family name → kernel function. ARD vs isotropic is encoded in
# ``theta`` length, not in a separate function.
_KERNEL_FAMILIES: dict[str, KernelFn] = {
    "rbf": rbf_kernel,
    "matern52": matern52_kernel,
}

#: Backwards-compatible alias mapping kernel names → callables. Both
#: ``"rbf"`` and ``"rbf_ard"`` resolve to :func:`rbf_kernel`; the difference
#: lives in the theta vector's length.
KERNELS: dict[KernelName, KernelFn] = {
    "rbf": rbf_kernel,
    "matern52": matern52_kernel,
    "rbf_ard": rbf_kernel,
    "matern52_ard": matern52_kernel,
}


def get_kernel(name: KernelName) -> KernelFn:
    """Lookup a registered kernel by name (raises ``ValueError`` if unknown)."""
    base = base_kernel_name(name)
    if base not in _KERNEL_FAMILIES:
        raise ValueError(
            f"Unknown kernel {name!r}; expected one of {list(KERNELS)}"
        )
    return _KERNEL_FAMILIES[base]


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

    noise = max(float(theta[-1]), 1e-12)
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
    noise = max(float(theta[-1]), 1e-12)
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
    enforcing positivity. We restart from random points in ``log_bounds`` and
    keep the best NLL.

    For ARD kernels the parameter vector grows from length 3 to ``2 + d``
    (one lengthscale per input dim). All lengthscales share the same prior
    bounds and starting distribution. With wide bounds and ``n_restarts >= 3``
    L-BFGS-B converges reliably for our :math:`d \\leq 10` problem.

    Args:
        x_train: ``(n, d)`` training inputs (z-scored upstream is recommended).
        y_train: ``(n,)`` training targets (centred recommended).
        kernel: One of :data:`KernelName` (``rbf`` / ``matern52`` /
            ``rbf_ard`` / ``matern52_ard``).
        n_restarts: Number of random restarts (in addition to
            ``init_log_theta`` and the canonical "all 1s" start).
        init_log_theta: Optional explicit starting point in log-space.
        seed: RNG seed for the restarts.
        bounds: Optional log-space bounds per parameter; defaults to a wide
            box ``[1e-3, 1e3]`` per dim. Must match the theta length if given.

    Returns:
        :class:`GPModel` with the best ``theta`` and cached training data.
    """
    if x_train.shape[0] == 0:
        raise ValueError("Cannot fit GP with zero training points")
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("x_train / y_train length mismatch")

    n_features = int(x_train.shape[1])
    n_l = n_lengthscales(kernel, n_features)
    n_params = 2 + n_l  # signal_var + lengthscales + noise_var
    kernel_fn = get_kernel(kernel)

    if bounds is None:
        bounds = ((np.log(1e-3), np.log(1e3)),) * n_params
    elif len(bounds) != n_params:
        raise ValueError(
            f"bounds has length {len(bounds)} but kernel {kernel!r} expects "
            f"theta length {n_params}"
        )

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    if init_log_theta is not None:
        init = np.asarray(init_log_theta, dtype=float)
        if init.size != n_params:
            raise ValueError(
                f"init_log_theta has length {init.size} but kernel {kernel!r} "
                f"expects theta length {n_params}"
            )
        starts.append(init)
    # Canonical start: signal=1, all_lengthscales=1, noise=0.1
    starts.append(
        np.concatenate([[0.0], np.zeros(n_l), [np.log(0.1)]])
    )
    for _ in range(max(0, n_restarts)):
        starts.append(
            np.concatenate(
                [
                    [rng.uniform(np.log(0.1), np.log(10.0))],
                    rng.uniform(np.log(0.3), np.log(10.0), size=n_l),
                    [rng.uniform(np.log(1e-3), np.log(1.0))],
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
